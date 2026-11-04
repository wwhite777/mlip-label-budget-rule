"""Frozen-backbone structure descriptors for WBM (P3, study).

One CLI:  python extract.py --backbone {mace_small,mace_medium,chgnet,mattersim,mattersim_5m,orb,sevenn,
                                       mace_small_randinit}
                            --ids <file: one material_id per line> --out <npz>
                            [--batch N] [--device cuda] [--mode batched|single] [--structures relaxed|initial]
                            [--energy --energy-out <npz>]

--energy (off by default; without it the feature path and the --out npz are exactly as before): the same forward
pass also returns the backbone's single-point TOTAL energy in eV per structure (no relaxation, no forces), saved to
--energy-out (material_id, energy_total, n_atoms, checkpoint sha256, failed ids). How each backbone's energy is
read from that pass is stated in ENERGY_INFO below. Features are computed before and independently of the energy
read-out (the energy is a head applied to the same tensors); mace --mode single does not support --energy.

--structures relaxed (default): relaxed WBM structures, cache/wbm/wbm_cse.json.bz2 (computed_structure_entry).
--structures initial: unrelaxed WBM inputs, cache/mbd/2022-10-19-wbm-init-structs.json.bz2 (initial_structure).
Failures are never dropped silently: if a batch raises, its structures are retried one by one; a structure that
still fails gets a NaN row (never zero-filled), its id and error go to failed_ids / failed_errors in the npz and
the log, and the process exits with code 3 after saving (exit 0 = no failures). CHGNet structures with isolated
atoms (no neighbour within 6 A) are kept and listed in isolated_atom_ids (npz) and the log.

Structure-level descriptor = mean over atoms of the backbone's final per-atom invariant
(scalar) features. Which tensor is pooled is stated in BACKBONE_INFO below.

Environments:
  ~/envs/analysis        mace_small, mace_medium, chgnet, orb   (e3nn 0.4.4)
  ~/envs/analysis_e3nn   sevenn, mattersim                      (e3nn 0.6.0)
"""
import argparse
import bz2
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
WBM_CSE = os.path.join(PROJECT, "cache", "wbm", "wbm_cse.json.bz2")
WBM_INIT = os.path.join(PROJECT, "cache", "mbd", "2022-10-19-wbm-init-structs.json.bz2")
STRUCT_FILES = {"relaxed": (WBM_CSE, "computed_structure_entry", "structure"),  # (file, key, sub-key)
                "initial": (WBM_INIT, "initial_structure", None)}
EXIT_FAILURES = 3

BACKBONE_INFO = {
    "mace_small": "MACE-MP-0 small (2023-12-10 L0 128ch): calc.get_descriptors(invariants_only=True, "
                  "all layers) = node_feats scalar 0e channels of both interaction layers concatenated; invariant",
    "mace_medium": "MACE-MP-0 medium (2023-12-03 L1 128ch): calc.get_descriptors(invariants_only=True, "
                   "all layers) = 0e channels of both layers concatenated; invariant",
    "chgnet": "CHGNet 0.3.0: atom_feas after the last atom-conv layer and readout LayerNorm "
              "(crystal_fea is their SUM when mlp_first; divided by n_atoms -> mean); invariant",
    "mattersim": "MatterSim-v1.0.0-1M (M3GNet): atom_attr after the last graph_conv = input to "
                 "model.final (forward hook); invariant",
    "orb": "ORB v3 conservative-inf-mpa (float32-highest): MoleculeGNS out['node_features'] after the last message-passing "
           "step (latent 256); not invariant by construction (SH edge features; v3 trained with "
           "rotational-equivariance regularisation)",
    "sevenn": "SevenNet-0 (11Jul2024): data['x'] after 4_equivariant_gate (irreps 128x0e, model truncated "
              "before reduce_input_to_hidden); invariant",
    "mattersim_5m": "MatterSim-v1.0.0-5M (M3GNet): atom_attr after the last graph_conv = input to "
                    "model.final (forward hook); invariant",
    "mace_small_randinit": "CONTROL: MACE-MP-0 small architecture/hyperparameters (mace.tools.scripts_utils."
                           "extract_config_mace_model of the 2023-12-10 L0 checkpoint), all trainable weights "
                           "re-initialised by the model's own constructor (ScaleShiftMACE(**config), float64, "
                           "torch.manual_seed(0), cast to float32); NO pretrained weight loaded; same extraction as "
                           "mace_small (0e channels of both layers, mean pool); invariant",
}

ENERGY_INFO = {  # --energy: how the single-point total energy (eV) is read from the same pass
    "mace_small": "out['energy'] of the same model(batch, compute_force=False) call that yields node_feats (float32)",
    "mace_medium": "out['energy'] of the same model(batch, compute_force=False) call that yields node_feats (float32)",
    "chgnet": "predict_graph(task='e') output 'e' (eV/atom) of the same call that yields crystal_fea, times n_atoms",
    "mattersim": "return value of the same model.forward(inp, -1) call (= Potential 'total_energy')",
    "mattersim_5m": "return value of the same model.forward(inp, -1) call (= Potential 'total_energy')",
    "orb": "energy head on the same node_features: heads['energy'].predict(node, batch) + ZBL pair repulsion "
           "(as ConservativeForcefieldRegressor.forward/predict; no force/stress autograd)",
    "sevenn": "the checkpoint's own tail modules reduce_input_to_hidden -> reduce_hidden_to_energy -> "
              "rescale_atomic_energy -> reduce_total_enegy applied to the same data after 4_equivariant_gate "
              "(force_output skipped); inferred_total_energy",
}


# ---------------------------------------------------------------- structures
def load_structures(mids, source="relaxed"):
    """WBM structures (pymatgen) for the given material_ids, in order; source 'relaxed' or 'initial'."""
    from pymatgen.core import Structure
    path, key, sub = STRUCT_FILES[source]
    with bz2.open(path, "rt") as fh:
        raw = json.load(fh)
    key_of = {mid: k for k, mid in raw["material_id"].items()}
    col = raw[key]
    return [Structure.from_dict(col[key_of[m]][sub] if sub else col[key_of[m]]) for m in mids]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _scatter_mean(x, batch, n_graphs):
    import torch
    out = torch.zeros(n_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    out.index_add_(0, batch, x)
    cnt = torch.bincount(batch, minlength=n_graphs).to(x.dtype).unsqueeze(1)
    return out / cnt


# ---------------------------------------------------------------- backbones
class MaceFeaturizer:
    def __init__(self, variant, device, mode="batched"):
        from mace.calculators import mace_mp
        from mace.calculators.foundations_models import download_mace_mp_checkpoint
        self.calc = mace_mp(model=variant, device=device, default_dtype="float32")
        self.weights = download_mace_mp_checkpoint(variant)  # already cached -> path only
        self.mode, self.device = mode, device
        self.needs_grad = mode == "single"  # get_descriptors runs the force head (autograd)
        self.want_energy, self.last_energy = False, None
        from pymatgen.io.ase import AseAtomsAdaptor
        self.adaptor = AseAtomsAdaptor()

    def _single(self, structures):
        out = []
        for s in structures:
            d = self.calc.get_descriptors(self.adaptor.get_atoms(s))  # exact original method
            out.append(d.mean(axis=0))
        return np.asarray(out, dtype=np.float32)

    def _batched(self, structures):
        """Same graphs and invariant extraction as calc.get_descriptors, many graphs per forward."""
        import torch
        from e3nn import o3
        from mace import data as mace_data
        from mace.modules.utils import extract_invariant
        from mace.tools import torch_geometric, torch_tools
        c = self.calc
        c.arrays_keys.update({c.charges_key: "charges"})
        keyspec = mace_data.KeySpecification(info_keys=c.info_keys, arrays_keys=c.arrays_keys)
        graphs = []
        with torch_tools.default_dtype(c.default_dtype):
            for s in structures:
                cfg = mace_data.config_from_atoms(self.adaptor.get_atoms(s), key_specification=keyspec,
                                                  head_name=c.head)
                graphs.append(mace_data.AtomicData.from_config(cfg, z_table=c.z_table, cutoff=c.r_max,
                                                               heads=c.available_heads))
        batch = torch_geometric.Batch.from_data_list(graphs).to(self.device)
        model = c.models[0]
        out = model(batch.to_dict(), compute_force=False, compute_stress=False)
        node = out["node_feats"]
        if self.want_energy:
            self.last_energy = out["energy"].detach().double().cpu().numpy() * float(c.energy_units_to_eV)
        n_int = int(model.num_interactions)
        irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        n_inv = irreps_out.dim // (l_max + 1) ** 2
        inv = extract_invariant(node, num_layers=n_int, num_features=n_inv, l_max=l_max)
        return _scatter_mean(inv.detach(), batch.batch, len(structures)).cpu().numpy()

    def featurize(self, structures):
        if self.want_energy and self.mode == "single":
            raise ValueError("--energy needs --mode batched for MACE")
        return self._single(structures) if self.mode == "single" else self._batched(structures)


class MaceRandinitFeaturizer(MaceFeaturizer):
    """Random-weight control: the calculator of the pretrained small model supplies graph construction (z_table,
    r_max, heads) and the hyperparameters; its model is replaced by a freshly constructed one (seed 0). The pretrained
    trainable weights never enter the new model; non-trainable buffers (atomic numbers, E0s, scale/shift, Bessel
    weights, avg_num_neighbors, r_max) come from the config, i.e. are hyperparameters."""
    SEED = 0

    def __init__(self, device, mode="batched"):
        import torch
        from mace.tools import torch_tools
        from mace.tools.scripts_utils import extract_config_mace_model
        super().__init__("small", device, mode)
        pre = self.calc.models[0]
        cfg = extract_config_mace_model(pre)
        with torch_tools.default_dtype(torch.float64):
            torch.manual_seed(self.SEED)
            model = type(pre)(**cfg)
        model = model.to(torch.float32).to(device).eval()
        for prm in model.parameters():
            prm.requires_grad = False  # as MACECalculator does for its models
        self.calc.models[0] = model
        h = hashlib.sha256()
        for k, v in model.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        self.weights_sha256 = h.hexdigest()  # of the random state_dict (keys + tensor bytes, float32)
        self.weights = (f"randinit:torch.manual_seed({self.SEED}) ScaleShiftMACE(**extract_config_mace_model(<{self.weights}>)) "
                        f"- architecture only, no pretrained weights")


class ChgnetFeaturizer:
    def __init__(self, device):
        import chgnet
        from chgnet.model import CHGNet
        self.model = CHGNet.load(model_name="0.3.0", use_device=device, verbose=False)
        self.model.eval()
        self.weights = os.path.join(os.path.dirname(chgnet.__file__), "pretrained", "0.3.0",
                                    "chgnet_0.3.0_e29f68s314m37.pth.tar")
        # chgnet 0.4.2 converter default is "error" (aborted the S0 full relaxed run); "warn" keeps the
        # structure, prints the converter message to stderr, and we flag it (last_isolated -> isolated_atom_ids)
        self.model.graph_converter.set_isolated_atom_response("warn")
        self.last_isolated = []
        self.want_energy, self.last_energy = False, None

    def featurize(self, structures):
        # same steps as CHGNet.predict_structure (convert each structure, then predict_graph)
        graphs = [self.model.graph_converter(s) for s in structures]
        # isolated atom = no outgoing edge in atom_graph (an all-isolated structure has atom_graph of shape (0,))
        self.last_isolated = [len(s) - len(set(g.atom_graph.reshape(-1, 2)[:, 0].tolist())) > 0
                              for g, s in zip(graphs, structures)]
        preds = self.model.predict_graph(graphs, task="e", return_crystal_feas=True, batch_size=len(structures))
        preds = [preds] if isinstance(preds, dict) else preds
        if self.want_energy:  # 'e' is eV/atom
            self.last_energy = np.asarray([float(p["e"]) * len(s) for p, s in zip(preds, structures)])
        return np.asarray([p["crystal_fea"] / len(s) for p, s in zip(preds, structures)], dtype=np.float32)


class OrbFeaturizer:
    URL = ("https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/"
           "orb-v3-conservative-inf-mpa-20250404.ckpt")

    def __init__(self, device):
        import torch
        from cached_path import cached_path
        from orb_models.forcefield import pretrained
        # float32-highest (no TF32): S0 smoke showed TF32 pass-to-pass |diff| ~1e-3 vs ~1e-6, same speed
        self.model = pretrained.orb_v3_conservative_inf_mpa(device=device, compile=False,
                                                            precision="float32-highest")
        self.weights = str(cached_path(self.URL))
        self.device = torch.device(device)
        from pymatgen.io.ase import AseAtomsAdaptor
        self.adaptor = AseAtomsAdaptor()
        self.want_energy, self.last_energy = False, None

    def featurize(self, structures):
        from orb_models.forcefield import atomic_system, base
        graphs = [atomic_system.ase_atoms_to_atom_graphs(self.adaptor.get_atoms(s), self.model.system_config,
                                                         device=self.device) for s in structures]
        batch = base.batch_graphs(graphs)
        vectors, _, _ = batch.compute_differentiable_edge_vectors()
        batch.edge_features["vectors"] = vectors
        node = self.model.model(batch)["node_features"]
        import torch
        n_node = batch.n_node.to(node.device)
        owner = torch.repeat_interleave(torch.arange(len(structures), device=node.device), n_node)
        feats = _scatter_mean(node.float(), owner, len(structures)).cpu().numpy()
        if self.want_energy:  # same steps as ConservativeForcefieldRegressor.forward + predict, minus autograd
            m = self.model
            e = m.heads[m.energy_name].predict(node, batch)
            if m.pair_repulsion:
                e = e + m.pair_repulsion_fn(batch)["energy"]
            self.last_energy = e.detach().double().cpu().numpy().reshape(-1)
        return feats


class SevennFeaturizer:
    def __init__(self, device):
        import sevenn.util as U
        self.cp = U.load_checkpoint("7net-0")
        self.weights = self.cp.checkpoint_path if hasattr(self.cp, "checkpoint_path") else \
            U.pretrained_name_to_path("7net-0")
        model = self.cp.build_model()
        keys = list(model._modules.keys())
        for k in keys[keys.index("4_equivariant_gate") + 1:]:
            model.delete_module_by_key(k)
        model.set_is_batch_data(True)
        self.model = model.to(device).eval()
        self.cutoff, self.device = float(self.cp.config["cutoff"]), device
        self.want_energy, self.last_energy, self._tail = False, None, None
        self._tail_keys = keys[keys.index("4_equivariant_gate") + 1:keys.index("force_output")]
        from pymatgen.io.ase import AseAtomsAdaptor
        self.adaptor = AseAtomsAdaptor()

    def featurize(self, structures):
        from sevenn.atom_graph_data import AtomGraphData
        from sevenn.train.dataload import unlabeled_atoms_to_graph
        from torch_geometric.data import Batch
        graphs = [AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(self.adaptor.get_atoms(s), self.cutoff))
                  for s in structures]
        batch = Batch.from_data_list(graphs).to(self.device)
        out = self.model(batch)
        feats = _scatter_mean(out["x"].detach(), out["batch"], len(structures)).cpu().numpy()
        if self.want_energy:
            if self._tail is None:  # the checkpoint's own energy modules, same weights
                full = self.cp.build_model()
                full.set_is_batch_data(True)
                self._tail = [full._modules[k].to(self.device).eval() for k in self._tail_keys]
            for mod in self._tail:
                out = mod(out)
            self.last_energy = out["inferred_total_energy"].detach().double().cpu().numpy().reshape(-1)
        return feats


class MattersimFeaturizer:
    def __init__(self, device, size="1M"):
        from mattersim.forcefield.potential import Potential
        self.pot = Potential.from_checkpoint(load_path=f"MatterSim-v1.0.0-{size}", device=device,
                                             load_training_state=False)
        self.pot.model.eval()
        self.weights = os.path.expanduser(f"~/.local/mattersim/pretrained_models/mattersim-v1.0.0-{size}.pth")
        self.want_energy, self.last_energy = False, None
        args = self.pot.model.model_args
        self.cutoff, self.tb_cutoff = float(args["cutoff"]), float(args["threebody_cutoff"])
        self.device, self._captured = device, None
        self.pot.model.final.register_forward_hook(lambda m, inp, out: setattr(self, "_captured", inp[0]))
        from pymatgen.io.ase import AseAtomsAdaptor
        self.adaptor = AseAtomsAdaptor()

    def featurize(self, structures):
        from mattersim.datasets.utils.build import build_dataloader
        from mattersim.forcefield.potential import batch_to_dict
        atoms = [self.adaptor.get_atoms(s) for s in structures]
        dl = build_dataloader(atoms, only_inference=True, batch_size=len(atoms), shuffle=False,
                              cutoff=self.cutoff, threebody_cutoff=self.tb_cutoff)
        outs, energies = [], []
        for gb in dl:
            inp = batch_to_dict(gb, device=self.device)
            e = self.pot.model.forward(inp, -1)
            outs.append(_scatter_mean(self._captured.detach(), inp["batch"], int(gb.num_graphs)).cpu().numpy())
            if self.want_energy:
                energies.append(e.detach().double().cpu().numpy().reshape(-1))
        if self.want_energy:
            self.last_energy = np.concatenate(energies)
        return np.concatenate(outs, axis=0)


def build_featurizer(name, device, mode="batched"):
    if name == "mace_small":
        return MaceFeaturizer("small", device, mode)
    if name == "mace_medium":
        return MaceFeaturizer("medium", device, mode)
    if name == "mattersim_5m":
        return MattersimFeaturizer(device, size="5M")
    if name == "mace_small_randinit":
        return MaceRandinitFeaturizer(device, mode)
    return {"chgnet": ChgnetFeaturizer, "orb": OrbFeaturizer, "sevenn": SevennFeaturizer,
            "mattersim": MattersimFeaturizer}[name](device)


def featurize_all(fz, structures, batch):
    import torch
    with torch.set_grad_enabled(getattr(fz, "needs_grad", False)):
        return np.concatenate([fz.featurize(c) for c in _chunks(structures, batch)], axis=0).astype(np.float32)


def _weights_sha256(fz):
    """sha256 of the checkpoint file; a random-init control carries the hash of its own state_dict instead."""
    return getattr(fz, "weights_sha256", None) or sha256_file(fz.weights)


def _take_energy(fz, n):
    """Energies of the last fz.featurize call (n structures), checked for length; None when --energy is off."""
    if not getattr(fz, "want_energy", False):
        return None
    e = np.asarray(fz.last_energy, dtype=np.float64).reshape(-1)
    if e.shape[0] != n:
        raise RuntimeError(f"energy read-out returned {e.shape[0]} values for {n} structures")
    return e


def featurize_robust(fz, structures, batch, mids):
    """featurize_all with per-structure retry; returns (feats, [(id, error), ...], isolated_atom_ids, energies).
    isolated_atom_ids: structures the backbone flagged as having isolated atoms (CHGNet only). Identical numbers to
    featurize_all when nothing fails (same batches, same calls, same concatenation). energies: float64 total
    energies (eV) from the same calls when fz.want_energy is set (NaN for failed structures), else None."""
    import torch
    parts, failures, flagged, e_parts = [], [], [], []
    with torch.set_grad_enabled(getattr(fz, "needs_grad", False)):
        for i0 in range(0, len(structures), batch):
            chunk = structures[i0:i0 + batch]
            try:
                p = fz.featurize(chunk)
                e_parts.append(_take_energy(fz, len(chunk)))
                parts.append(p)
                flagged += [mids[i0 + j] for j, f in enumerate(getattr(fz, "last_isolated", [])) if f]
            except Exception as e:  # noqa: BLE001 - recorded, never silent
                print(f"[extract] batch starting at row {i0} failed ({type(e).__name__}: {e}); retrying "
                      f"structure by structure", flush=True)
                for j, s in enumerate(chunk):
                    try:
                        p = fz.featurize([s])
                        e_parts.append(_take_energy(fz, 1))
                        parts.append(p)
                        if any(getattr(fz, "last_isolated", [])):
                            flagged.append(mids[i0 + j])
                    except Exception as e2:  # noqa: BLE001
                        failures.append((mids[i0 + j], f"{type(e2).__name__}: {e2}"))
                        print(f"[extract] FAILED id={mids[i0 + j]}: {type(e2).__name__}: {e2}", flush=True)
                        parts.append(None)
                        e_parts.append(np.full(1, np.nan))
    dims = {p.shape[1] for p in parts if p is not None}
    if len(dims) != 1:
        raise RuntimeError(f"no successful structure or inconsistent dims {dims}")
    d = dims.pop()
    parts = [np.full((1, d), np.nan, np.float32) if p is None else p for p in parts]
    energies = np.concatenate(e_parts) if getattr(fz, "want_energy", False) else None
    return np.concatenate(parts, axis=0).astype(np.float32), failures, flagged, energies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONE_INFO))
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="batched", choices=["batched", "single"], help="MACE only")
    ap.add_argument("--structures", default="relaxed", choices=sorted(STRUCT_FILES))
    ap.add_argument("--energy", action="store_true", help="also store single-point total energies (eV)")
    ap.add_argument("--energy-out", default=None, help="npz for --energy (required with --energy)")
    a = ap.parse_args()
    if a.energy and not a.energy_out:
        ap.error("--energy needs --energy-out")
    import torch
    mids = [ln.strip() for ln in open(a.ids) if ln.strip()]
    t0 = time.time()
    structures = load_structures(mids, a.structures)
    t_load = time.time() - t0
    fz = build_featurizer(a.backbone, a.device, a.mode)
    if a.energy:
        fz.want_energy = True
    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    sync()
    t1 = time.time()
    feats, failures, isolated, energies = featurize_robust(fz, structures, a.batch, mids)
    sync()
    t_feat = time.time() - t1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, feats=feats, mids=np.array(mids), backbone=a.backbone,
                        pooled_tensor=BACKBONE_INFO[a.backbone], weights_path=str(fz.weights),
                        weights_sha256=_weights_sha256(fz), seconds_featurize=t_feat,
                        structures_source=a.structures, structures_file=STRUCT_FILES[a.structures][0],
                        structures_sha256=sha256_file(STRUCT_FILES[a.structures][0]),
                        failed_ids=np.array([f[0] for f in failures], dtype=str),
                        failed_errors=np.array([f[1] for f in failures], dtype=str),
                        isolated_atom_ids=np.array(isolated, dtype=str))
    if a.energy:
        os.makedirs(os.path.dirname(os.path.abspath(a.energy_out)), exist_ok=True)
        np.savez_compressed(a.energy_out, material_id=np.array(mids), energy_total=energies,
                            n_atoms=np.array([len(s) for s in structures], dtype=np.int64), backbone=a.backbone,
                            energy_source=ENERGY_INFO[a.backbone], energy_unit="eV (total, per structure)",
                            weights_path=str(fz.weights), weights_sha256=_weights_sha256(fz),
                            structures_source=a.structures, structures_file=STRUCT_FILES[a.structures][0],
                            structures_sha256=sha256_file(STRUCT_FILES[a.structures][0]),
                            seconds_pass=t_feat, features_npz=os.path.abspath(a.out),
                            failed_ids=np.array([f[0] for f in failures], dtype=str),
                            failed_errors=np.array([f[1] for f in failures], dtype=str))
        print(f"[extract] energy: n={len(energies)} non-finite={int((~np.isfinite(energies)).sum())} -> "
              f"{a.energy_out}", flush=True)
    pos = {m: i for i, m in enumerate(mids)}
    for m in isolated:
        print(f"[extract] ISOLATED-ATOM structure id={m} (kept; row finite={bool(np.isfinite(feats[pos[m]]).all())})",
              flush=True)
    print(f"[extract] {a.backbone} n={len(mids)} dim={feats.shape[1]} load_structs={t_load:.1f}s "
          f"featurize={t_feat:.1f}s ({len(mids) / t_feat:.2f} structures/s incl. first-batch warm-up) "
          f"structures={a.structures} failed={len(failures)} isolated_atom={len(isolated)} -> {a.out}", flush=True)
    if failures:
        sys.exit(EXIT_FAILURES)


if __name__ == "__main__":
    main()
