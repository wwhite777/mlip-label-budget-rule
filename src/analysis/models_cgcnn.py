"""CGCNN control from scratch (torch_geometric), adapted from papers/p3_materials_fm/method_p3.py with the YAML grid,
optimizer, batch size, max epochs and early stopping on inner-validation average precision.

Graph (INITIAL structures, YAML controls.cgcnn.graph): nodes = sites, node input = one-hot atomic number (Z <= 100,
then a linear embedding); for each site its <= max_neighbors nearest neighbours within cutoff_angstrom (pymatgen
get_neighbor_list, periodic images included, ties by distance then neighbour index), messages neighbour -> site;
edge feature = 16 Gaussians on [0, cutoff] (method_p3.py). A site with no neighbour inside the cutoff gets a self-loop
at distance 0 (method_p3.py fallback; counted).
Model: Linear(100, hidden) -> n_conv x [CGConv(hidden, dim=16, batch_norm=True) + ReLU] -> mean pool ->
Linear(hidden, hidden) -> ReLU -> Linear(hidden, 1); unweighted BCE-with-logits; minibatches of batch_size in a
numpy default_rng(seed + epoch) permutation of the training set; torch.manual_seed(seed) before the model is built,
seed = the replicate's draw seed (YAML budgets.draw: 1000*fold + replicate) for every grid point (paired configs).
Determinism: the caller decides the device; run.py / cgcnn_timing.py record device and determinism mode.
Early stopping: after every epoch the inner-validation AP is computed; the best (first maximum) epoch is kept; training
stops after `patience` epochs without improvement or at max_epochs. Refit on all n labels for exactly best_epoch epochs.
"""
from __future__ import annotations

import bz2
import json
import time

import numpy as np

from analysis.config import AnalysisError

Z_MAX = 100
N_GAUSS = 16


def load_initial_structures(path, ids) -> list:
    """pymatgen Structures of the given material_ids from the initial-structure file (one json load)."""
    from pymatgen.core import Structure
    with bz2.open(path, "rt") as fh:
        raw = json.load(fh)
    key_of = {mid: k for k, mid in raw["material_id"].items()}
    col = raw["initial_structure"]
    return [Structure.from_dict(col[key_of[m]]) for m in ids]


def build_graph(structure, cutoff: float, max_nbr: int) -> dict:
    c, p, _, d = structure.get_neighbor_list(float(cutoff))
    z = np.array([min(int(sp.Z), Z_MAX) for sp in structure.species], dtype=np.int64)
    if len(c):
        o = np.lexsort((p, d, c))
        c, p, d = c[o], p[o], d[o]
        first = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
        rank = np.arange(len(c)) - np.repeat(first, np.diff(np.r_[first, len(c)]))
        k = rank < int(max_nbr)
        c, p, d = c[k], p[k], d[k]
    isolated = sorted(set(range(len(z))) - set(c.tolist()))
    if isolated:
        c = np.r_[c, isolated].astype(np.int64)
        p = np.r_[p, isolated].astype(np.int64)
        d = np.r_[d, np.zeros(len(isolated))]
    return {"z": z, "src": p.astype(np.int64), "dst": c.astype(np.int64), "dist": d.astype(np.float32),
            "n_isolated": len(isolated)}


def build_graphs(structures, cutoff, max_nbr, workers: int = 1) -> list:
    if workers <= 1:
        return [build_graph(s, cutoff, max_nbr) for s in structures]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(build_graph, structures, [cutoff] * len(structures), [max_nbr] * len(structures),
                           chunksize=256))


def _to_data(g):
    import torch
    from torch_geometric.data import Data
    return Data(z=torch.from_numpy(g["z"]), edge_index=torch.from_numpy(np.stack([g["src"], g["dst"]])),
                dist=torch.from_numpy(g["dist"]), num_nodes=len(g["z"]))


def _model(hidden: int, n_conv: int, cutoff: float):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import CGConv, global_mean_pool

    class CGCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(Z_MAX, hidden)
            self.convs = nn.ModuleList([CGConv(hidden, dim=N_GAUSS, batch_norm=True) for _ in range(n_conv)])
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            centers = torch.linspace(0.0, float(cutoff), N_GAUSS)
            self.register_buffer("centers", centers)
            self.width = float(centers[1] - centers[0])

        def forward(self, b):
            x = self.embed(torch.nn.functional.one_hot(b.z - 1, Z_MAX).float())
            ea = torch.exp(-((b.dist[:, None] - self.centers[None, :]) ** 2) / self.width ** 2)
            for conv in self.convs:
                x = torch.relu(conv(x, b.edge_index, ea))
            return self.head(global_mean_pool(x, b.batch)).squeeze(-1)

    return CGCNN()


class CGCNNControl:
    def __init__(self, params: dict, spec: dict, device: str = "cpu", seed: int = 0):
        if spec.get("optimizer") != "AdamW" or spec["early_stopping"]["metric"] != "inner_validation_average_precision":
            raise AnalysisError("cgcnn spec changed")
        self.p, self.spec, self.device, self.seed = params, spec, device, int(seed)
        self.cutoff = float(spec["graph"]["cutoff_angstrom"])

    def _batches(self, data, y, order):
        import torch
        from torch_geometric.data import Batch
        bs = int(self.spec["batch_size"])
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            b = Batch.from_data_list([data[j] for j in idx]).to(self.device)
            yield b, (None if y is None else torch.as_tensor(y[idx], dtype=torch.float32, device=self.device))

    def _new(self):
        import torch
        torch.manual_seed(self.seed)
        m = _model(int(self.p["hidden"]), int(self.p["n_conv"]), self.cutoff).to(self.device)
        opt = torch.optim.AdamW(m.parameters(), lr=float(self.p["lr"]), weight_decay=float(self.spec["weight_decay"]))
        return m, opt

    def _epoch(self, m, opt, data, y, ep):
        import torch
        crit = torch.nn.BCEWithLogitsLoss()
        m.train()
        order = np.random.default_rng(self.seed + ep).permutation(len(data))
        for b, yb in self._batches(data, y, order):
            opt.zero_grad(set_to_none=True)
            crit(m(b), yb).backward()
            opt.step()

    def _predict(self, m, data):
        import torch
        m.eval()
        out = []
        with torch.no_grad():
            for b, _ in self._batches(data, None, np.arange(len(data))):
                out.append(m(b).double().cpu().numpy())
        z = np.concatenate(out) if out else np.zeros(0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit_early_stopping(self, g_tr, y_tr, g_va, y_va, ap_fn):
        """Returns (best inner-val AP, best_epoch, history, seconds)."""
        t0 = time.time()
        d_tr, d_va = [_to_data(g) for g in g_tr], [_to_data(g) for g in g_va]
        m, opt = self._new()
        best, best_ep, hist = -1.0, 0, []
        pat, max_ep = int(self.spec["early_stopping"]["patience"]), int(self.spec["max_epochs"])
        for ep in range(1, max_ep + 1):
            self._epoch(m, opt, d_tr, np.asarray(y_tr), ep)
            ap = float(ap_fn(np.asarray(y_va), self._predict(m, d_va)))
            hist.append(ap)
            if ap > best:
                best, best_ep = ap, ep
            elif ep - best_ep >= pat:
                break
        return best, best_ep, hist, time.time() - t0

    def refit(self, g_all, y_all, epochs: int):
        t0 = time.time()
        d = [_to_data(g) for g in g_all]
        self.m, opt = self._new()
        for ep in range(1, int(epochs) + 1):
            self._epoch(self.m, opt, d, np.asarray(y_all), ep)
        self.refit_seconds = time.time() - t0
        return self

    def score(self, g_test):
        return self._predict(self.m, [_to_data(g) for g in g_test])
