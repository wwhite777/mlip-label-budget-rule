"""S5 timing probe at CONFIRM scale on SYNTHETIC arrays (no CONFIRM structure, feature or label is read).

python -m analysis.s5_probe --out result/s5_dryrun/probe_confirm_scale.json --device cuda
Shapes from the YAML partition block: N = confirm_structures; a fold = N/n_folds test, the rest training; the coverage
reference sample size, k, UMAP parameters and the C5 schedule / MLP from the v1.2 YAML. Times:
  knn_<d>        coverage k-NN of one fold (test -> training, reference sample -> training leave-self-out), float64
  umap_kde       UMAP fit on the reference sample + transform of the test fold + KDE (one random_state), d = 256
  mlp_fit_<n>    one MLP-ensemble fit (frozen hyperparameters) at n labels, d = 256; pool scoring of the training fold
  boot_row_draw  one bootstrap row-draw (weighted AP; AP + AUROC) over N structures
python -m analysis.s5_probe --units --out result/s5_dryrun/probe_units_pilot.json --device cuda
  per-budget unit components on the PILOT part with SYNTHETIC labels (dryrun configuration; fold 0, replicate 0):
  heads menu per pilot backbone, residual menu (stability), control menu without CGCNN, at every n of budgets.all_n.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analysis.s5_probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="as analysis.run (threadpool_limits + torch threads)")
    a = ap.parse_args(argv)
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(a.threads)
    with threadpool_limits(a.threads):
        if a.units:
            return units(a)
        return confirm_scale(a)


def confirm_scale(a) -> int:
    import torch

    from analysis import coverage as COV
    from analysis import metrics as Mx
    from analysis.config import c5_spec, coverage_spec, load_prereg
    from analysis.models import MLPEnsemble
    from analysis.outputs import write_json
    y = load_prereg()
    N = int(y["population"]["partition"]["confirm_structures"])
    nf = int(y["splits"]["n_folds"])
    n_test = N // nf
    n_train = N - n_test
    cs, c5 = coverage_spec(y), c5_spec(y)
    rng = np.random.default_rng(0)
    out = {"threads": a.threads, "synthetic": True, "N_confirm": N, "n_train_fold": n_train, "n_test_fold": n_test, "device": a.device,
           "torch": torch.__version__, "timings_s": {}, "loadavg_start": list(os.getloadavg())}
    T = out["timings_s"]
    if a.device.startswith("cuda"):
        out["gpu_name"] = torch.cuda.get_device_name(0)
    for d in (256, 89):
        Xtr = rng.standard_normal((n_train, d))
        Xte = rng.standard_normal((n_test, d)) + 0.1
        pos = rng.choice(n_train, size=cs["n_reference"], replace=False)
        t0 = time.time()
        r1 = COV.knn(Xte, Xtr, cs["k"], a.device)
        t1 = time.time()
        r2 = COV.knn(Xtr[pos], Xtr, cs["k"], a.device, self_idx=pos)
        t2 = time.time()
        chk = COV.cpu_crosscheck(Xte, Xtr, cs["k"], r1, np.arange(64))
        t3 = time.time()
        T[f"knn_d{d}_test"] = round(t1 - t0, 2)
        T[f"knn_d{d}_reference"] = round(t2 - t1, 2)
        T[f"knn_d{d}_crosscheck_64rows"] = round(t3 - t2, 2)
        out[f"knn_d{d}_crosscheck"] = chk
        out[f"knn_d{d}_fallback_rows"] = r1["n_fallback_rows"] + r2["n_fallback_rows"]
        if d == 256:
            t0 = time.time()
            u = COV.umap_kde(Xtr[pos], Xte, cs, 0)
            T["umap_kde_d256_one_random_state"] = round(time.time() - t0, 2)
            out["umap_kde_seconds_internal"] = u["seconds"]
        del Xtr, Xte
    mlp = dict(y["heads"]["mlp_ensemble"])
    X = rng.standard_normal((n_train, 256)).astype(np.float32)
    for n in (c5["initial"], c5["final"], int(y["budgets"]["primary_n"])):
        lab = rng.choice(n_train, size=n, replace=False)
        yy = (rng.random(n) < 0.2).astype(np.int8)
        t0 = time.time()
        m = MLPEnsemble(mlp, a.device).fit(X[lab], yy)
        t1 = time.time()
        m.member_probs(X)
        t2 = time.time()
        T[f"mlp_fit_n{n}"] = round(t1 - t0, 2)
        T[f"mlp_member_probs_pool_n{n}"] = round(t2 - t1, 2)
    s = rng.random(N)
    yy = (rng.random(N) < 0.2).astype(np.int8)
    ss = Mx.SortedScore(s, yy)
    sys_index = np.sort(rng.integers(0, int(y["population"]["partition"]["confirm_systems"]), size=N))
    S = int(sys_index.max()) + 1
    reps = 20
    W = [np.bincount(rng.integers(0, S, size=S), minlength=S)[sys_index] for _ in range(reps)]
    t0 = time.time()
    for w in W:
        ss.ap(w)
    t1 = time.time()
    for w in W:
        ss.ap(w)
        ss.auroc(w)
    t2 = time.time()
    for _ in range(reps):
        np.bincount(rng.integers(0, S, size=S), minlength=S)[sys_index]
    t3 = time.time()
    T["boot_row_draw_ap"] = round((t1 - t0) / reps, 5)
    T["boot_row_draw_ap_auroc"] = round((t2 - t1) / reps, 5)
    T["boot_draw_weights"] = round((t3 - t2) / reps, 5)
    out["loadavg_end"] = list(os.getloadavg())
    print(json.dumps(out, indent=1))
    write_json(a.out, out)
    return 0


def units(a) -> int:
    import torch

    from analysis.config import Config
    from analysis.draws import draw_labels, inner_split, outer_train_test
    from analysis.outputs import write_json
    from analysis.pipeline import load_inputs
    from analysis.selection import select_and_refit
    torch.use_deterministic_algorithms(True)
    y0 = Config.build("dryrun").y
    all_n = [int(n) for n in y0["budgets"]["all_n"]]
    cfg = Config.build("dryrun", budgets=all_n)
    inp = load_inputs(cfg, log=lambda *x: None)
    part, y = inp.part, cfg.y
    out = {"threads": a.threads, "synthetic_labels": True, "part": cfg.part, "N": part.n, "device": a.device, "fold": 0, "replicate": 0,
           "backbones": {bb: int(inp.feats[bb].shape[1]) for bb in cfg.backbones}, "seconds": {},
           "loadavg_start": list(os.getloadavg())}
    from analysis.data import LabelVault
    for t in cfg.tasks:
        vault = LabelVault(inp.labels[t], t)
        for n in all_n:
            draw = draw_labels(cfg, part, 0, 0, n)
            tr, va = inner_split(cfg, part, draw, 0, 0)
            _, test = outer_train_test(part, 0)
            view = vault.open((t, n), draw)
            rec = {}
            for bb in cfg.backbones:
                t0 = time.time()
                select_and_refit(cfg.heads, list(y["heads"]["tie_order"]), {h: cfg.head_grid(h) for h in cfg.heads},
                                 {h: inp.feats[bb] for h in cfg.heads}, view, tr, va, draw, test, y, a.device)
                rec[f"heads:{bb}"] = round(time.time() - t0, 2)
                if t == "stability":
                    Xr = np.hstack([inp.feats[bb].astype(np.float64), inp.refs[bb]["r_sp"]["e_hull"][:, None]])
                    t0 = time.time()
                    for m in cfg.residual_models():
                        select_and_refit([m], [m], {m: cfg.head_grid(m)}, {m: Xr}, view, tr, va, draw, test, y,
                                         a.device)
                    rec[f"residual:{bb}"] = round(time.time() - t0, 2)
            ctl = cfg.controls_at(n, t)
            t0 = time.time()
            select_and_refit(ctl, list(y["controls"]["tie_order"]), {c: cfg.control_grid(c) for c in ctl},
                             inp.control_X, view, tr, va, draw, test, y, a.device)
            rec["controls"] = round(time.time() - t0, 2)
            vault.close((t, n))
            out["seconds"][f"{t}_n{n}"] = rec
            print(t, n, rec, flush=True)
    out["loadavg_end"] = list(os.getloadavg())
    write_json(a.out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
