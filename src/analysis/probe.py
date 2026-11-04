"""Timing / determinism probes on SYNTHETIC labels (no real label column is read).

python -m analysis.probe unit  --out F.json [--device cuda]   one PILOT unit (n = pilot n, fold 0, replicate 0, both
                                                              tasks) run twice: wall time and bitwise score equality
python -m analysis.probe cgcnn --out F.json [--device cuda|cpu] [--config-index I] [--max-epochs E]
    CONFIRM fold 0, replicate 0, n = 3,000, one CGCNN grid point, band-gap synthetic labels: graph build time,
    early-stopping fit (inner train / inner validation), refit for best_epoch, scoring of the 34.5k test fold; the
    fit is run twice with identical seeds and the inner-validation AP history and test scores compared bitwise.
    --max-epochs E (probe only) fixes the epoch count of both runs (no early stopping) for a per-epoch cost.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _unit(a):
    import numpy as np
    import torch
    from analysis.config import Config
    from analysis.pipeline import State, load_inputs, run_unit
    torch.use_deterministic_algorithms(True)
    cfg = Config.build("dryrun", ["stability", "band_gap"])
    inp = load_inputs(cfg)
    res = {"device": a.device, "n": cfg.primary_n, "fold": 0, "replicate": 0, "runs": []}
    scores = []
    for k in range(2):
        st = State(cfg, inp)
        t0 = time.time()
        run_unit(cfg, inp, st, cfg.primary_n, 0, 0, a.device, None)
        res["runs"].append({"seconds": round(time.time() - t0, 1),
                            "unit_seconds": {u: v["seconds"] for u, v in st.units.items()}})
        scores.append({f"{t}|{m}": A[0].copy() for (t, n), d in st.oof.items() for m, A in d.items()})
    same = {k: bool(np.array_equal(scores[0][k], scores[1][k], equal_nan=True)) for k in scores[0]}
    res["bitwise_identical"] = all(same.values())
    res["per_method_identical"] = same
    if a.device.startswith("cuda"):
        res["max_memory_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    return res


def _cgcnn(a):
    import numpy as np
    import torch
    from analysis import metrics as Mx
    from analysis.config import Config, check_sha, resolve
    from analysis.data import load_part, synthetic_labels
    from analysis.draws import draw_labels, inner_split, outer_train_test
    from analysis.models_cgcnn import CGCNNControl, build_graphs, load_initial_structures
    cfg = Config.build("confirm", ["band_gap"])
    part = load_part(cfg)
    y = synthetic_labels(part)[0]["band_gap"]
    f, r, n = 0, 0, cfg.primary_n
    draw = draw_labels(cfg, part, f, r, n)
    tr, va = inner_split(cfg, part, draw, f, r)
    _, test = outer_train_test(part, f)
    spec = cfg.y["controls"]["cgcnn"]
    grid = cfg.control_grid("cgcnn")
    params = grid[a.config_index]
    res = {"device": a.device, "part": part.n, "n": n, "n_inner_train": len(tr), "n_inner_val": len(va),
           "n_test": len(test), "params": params, "grid_size": len(grid), "torch": torch.__version__,
           "threads": torch.get_num_threads()}
    ini = cfg.y["population"]["initial_structures"]
    path = resolve(ini["path"])
    check_sha(path, ini["sha256"], "initial structures")
    need = np.r_[draw, test]
    t0 = time.time()
    structs = load_initial_structures(path, part.ids[need])
    res["seconds_load_structures"] = round(time.time() - t0, 1)
    t0 = time.time()
    g = spec["graph"]
    graphs = build_graphs(structs, g["cutoff_angstrom"], g["max_neighbors"], a.graph_workers)
    res["seconds_build_graphs"] = round(time.time() - t0, 1)
    res["graphs_per_second"] = round(len(graphs) / max(res["seconds_build_graphs"], 1e-9), 1)
    res["n_graphs"] = len(graphs)
    res["n_isolated_sites"] = int(sum(x["n_isolated"] for x in graphs))
    gi = {int(i): x for i, x in zip(need, graphs)}

    def G(idx):
        return [gi[int(i)] for i in idx]

    if a.max_epochs:
        spec = dict(spec, max_epochs=a.max_epochs, early_stopping=dict(spec["early_stopping"], patience=10 ** 6))
    seed = cfg.draw_rng[0] * f + r + cfg.draw_rng[1]
    torch.use_deterministic_algorithms(True)
    res["deterministic_algorithms"] = True
    runs = []
    for k in range(2):
        rec = {}
        try:
            ctl = CGCNNControl(params, spec, a.device, seed)
            ap, ep, hist, secs = ctl.fit_early_stopping(G(tr), y[tr], G(va), y[va], Mx.ap)
            rec.update(best_ap=ap, best_epoch=ep, epochs_run=len(hist), seconds_fit=round(secs, 1), history=hist)
            ctl.refit(G(draw), y[draw], ep)
            rec["seconds_refit"] = round(ctl.refit_seconds, 1)
            t0 = time.time()
            s = ctl.score(G(test))
            rec["seconds_score"] = round(time.time() - t0, 1)
            rec["test_score_sha"] = __import__("hashlib").sha256(s.tobytes()).hexdigest()
            rec["_scores"] = s
        except RuntimeError as e:  # e.g. an op without a deterministic CUDA implementation
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        runs.append(rec)
        if "error" in rec:
            break
    ok = len(runs) == 2 and all("error" not in x for x in runs)
    res["bitwise_identical"] = bool(ok and runs[0]["history"] == runs[1]["history"]
                                    and np.array_equal(runs[0]["_scores"], runs[1]["_scores"]))
    for x in runs:
        x.pop("_scores", None)
    res["runs"] = runs
    if a.device.startswith("cuda"):
        res["max_memory_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    if ok:
        x = runs[-1]  # the second (warm) run
        ep_fit = x["seconds_fit"] / x["epochs_run"]
        ep_refit = x["seconds_refit"] / max(x["best_epoch"], 1)
        res["per_epoch_fit_plus_val_s"] = round(ep_fit, 3)
        res["per_epoch_refit_s"] = round(ep_refit, 3)
        R, K = int(cfg.y["replicates"]["confirm"]), cfg.n_folds
        worst_unit = len(grid) * int(cfg.y["controls"]["cgcnn"]["max_epochs"]) * ep_fit + \
            int(cfg.y["controls"]["cgcnn"]["max_epochs"]) * ep_refit + x["seconds_score"]
        res["estimate"] = {
            "assumption": "every grid point runs max_epochs (no early stop) = upper bound; graph build for the "
                          "whole CONFIRM part at the measured rate; per task",
            "graph_build_all_confirm_s": round(part.n / res["graphs_per_second"], 0),
            "per_fold_replicate_worst_s": round(worst_unit, 0),
            "per_task_worst_h": round(K * R * worst_unit / 3600, 2),
            "both_tasks_worst_h": round(2 * K * R * worst_unit / 3600, 2)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analysis.probe")
    ap.add_argument("what", choices=["unit", "cgcnn"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--graph-workers", type=int, default=8)
    ap.add_argument("--config-index", type=int, default=7)
    ap.add_argument("--max-epochs", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    if a.device.startswith("cuda"):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not vis or "0" in [v.strip() for v in vis.split(",")]:
            raise SystemExit("refusing: set CUDA_VISIBLE_DEVICES to GPUs among 1-3")
    torch.set_num_threads(a.threads)
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    with threadpool_limits(a.threads):
        res = _unit(a) if a.what == "unit" else _cgcnn(a)
    res["wall_s"] = round(time.time() - t0, 1)
    res["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    res["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items() if k not in ("runs", "per_method_identical")}, sort_keys=True))


if __name__ == "__main__":
    main()
