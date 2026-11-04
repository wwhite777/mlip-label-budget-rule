"""Learners of the head and control menus (hyperparameters from the YAML; see config.head_grid / control_grid).

logistic      StandardScaler (training statistics) + sklearn LogisticRegression(C, class_weight, max_iter); lbfgs.
hist_gbm      sklearn HistGradientBoostingClassifier(learning_rate, max_leaf_nodes, max_iter, early_stopping=False,
              random_state=0); no scaling (trees are invariant to the affine standardisation).
mlp_ensemble  StandardScaler + `members` MLPs d -> hidden... -> 1 (ReLU, dropout), AdamW(lr, weight_decay), `epochs`
              full-batch epochs of unweighted BCE-with-logits; member m seeded torch.manual_seed(m); score = mean
              member probability (sigmoid in float64).
Every learner returns a float64 score (higher = more likely positive).
"""
from __future__ import annotations

import numpy as np

from analysis.config import AnalysisError


class Logistic:
    def __init__(self, C: float, spec: dict):
        self.C, self.spec = float(C), spec

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        cw = self.spec.get("class_weight", None)
        self.m = LogisticRegression(C=self.C, class_weight=cw, max_iter=int(self.spec.get("max_iter", 2000)))
        self.m.fit(self.sc.transform(X), y)
        self.n_iter = int(np.max(self.m.n_iter_))
        return self

    def score(self, X):
        return self.m.predict_proba(self.sc.transform(X))[:, 1].astype(np.float64)


class HistGBM:
    def __init__(self, learning_rate: float, max_leaf_nodes: int, max_iter: int):
        self.p = dict(learning_rate=float(learning_rate), max_leaf_nodes=int(max_leaf_nodes), max_iter=int(max_iter))

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(**self.p, early_stopping=False, random_state=0).fit(X, y)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1].astype(np.float64)


class MLPEnsemble:
    def __init__(self, spec: dict, device: str = "cpu"):
        if spec.get("optimizer") != "AdamW":
            raise AnalysisError("mlp optimizer is not AdamW")
        self.spec, self.device = spec, device

    def _net(self, d):
        import torch.nn as nn
        layers, prev = [], d
        for h in self.spec["hidden"]:
            layers += [nn.Linear(prev, int(h)), nn.ReLU(), nn.Dropout(float(self.spec["dropout"]))]
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        Xt = torch.as_tensor(self.sc.transform(X), dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        crit = torch.nn.BCEWithLogitsLoss()
        self.members = []
        for m in range(int(self.spec["members"])):
            torch.manual_seed(m)
            net = self._net(Xt.shape[1]).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=float(self.spec["lr"]),
                                    weight_decay=float(self.spec["weight_decay"]))
            net.train()
            for _ in range(int(self.spec["epochs"])):
                opt.zero_grad(set_to_none=True)
                loss = crit(net(Xt).squeeze(-1), yt)
                loss.backward()
                opt.step()
            net.eval()
            self.members.append(net)
        return self

    def score(self, X):
        import torch
        Xt = torch.as_tensor(self.sc.transform(X), dtype=torch.float32, device=self.device)
        out = np.zeros(len(X), dtype=np.float64)
        with torch.no_grad():
            for net in self.members:
                z = net(Xt).squeeze(-1).double().cpu().numpy()
                out += 1.0 / (1.0 + np.exp(-z))
        return out / len(self.members)

    def member_probs(self, X) -> np.ndarray:
        """(members, len(X)) float64 member probabilities, each computed exactly as in score() (C5 uncertainty =
        their standard deviation). score(X) == the sequential sum of these rows / members (bit for bit)."""
        import torch
        Xt = torch.as_tensor(self.sc.transform(X), dtype=torch.float32, device=self.device)
        out = np.empty((len(self.members), len(X)), dtype=np.float64)
        with torch.no_grad():
            for i, net in enumerate(self.members):
                z = net(Xt).squeeze(-1).double().cpu().numpy()
                out[i] = 1.0 / (1.0 + np.exp(-z))
        return out


def make(kind: str, params: dict, y: dict, device: str = "cpu"):
    """kind in {logistic, hist_gbm, mlp_ensemble, composition_logistic, composition_hist_gbm,
    random_projection_logistic, randinit_encoder_logistic}; params = one grid point."""
    if kind in ("logistic", "composition_logistic", "random_projection_logistic", "randinit_encoder_logistic"):
        spec = dict(y["heads"]["logistic"])  # class_weight / max_iter of the logistic head
        return Logistic(params["C"], spec)
    if kind in ("hist_gbm", "composition_hist_gbm"):
        spec = y["heads"]["hist_gbm"] if kind == "hist_gbm" else y["controls"]["composition_hist_gbm"]
        return HistGBM(params["learning_rate"], params["max_leaf_nodes"], spec["max_iter"])
    if kind == "mlp_ensemble":
        return MLPEnsemble(y["heads"]["mlp_ensemble"], device)
    raise AnalysisError(f"unknown learner {kind}")
