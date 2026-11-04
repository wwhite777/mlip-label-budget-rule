"""Check the --energy read-out of src/featurize/extract.py against each package's own ASE / predict calculator.

python check_energy_pass.py --backbone <b> --ids <file> --n 50 --out <json>
Same initial structures, same checkpoint; the official path is the package's documented single-point call
(MACECalculator / CHGNet.predict_structure / ORBCalculator / SevenNetCalculator / MatterSimCalculator).
Writes max |E_extract - E_official| (eV/atom and eV) and the per-structure values. Energies only, no labels.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "featurize"))
import extract as E  # noqa: E402


def official_energies(b, fz, structures, device):
    from pymatgen.io.ase import AseAtomsAdaptor
    ad = AseAtomsAdaptor()
    if b.startswith("mace"):
        calc = fz.calc  # the MACECalculator the featurizer was built from (float32, same checkpoint)
        out = []
        for s in structures:
            at = ad.get_atoms(s)
            at.calc = calc
            out.append(at.get_potential_energy())
        return np.array(out), "mace.calculators.mace_mp(...).get_potential_energy (float32)"
    if b == "chgnet":
        return (np.array([float(fz.model.predict_structure(s, task="e")["e"]) * len(s) for s in structures]),
                "CHGNet.predict_structure(task='e')['e'] * n_atoms")
    if b == "orb":
        from orb_models.forcefield.calculator import ORBCalculator
        calc = ORBCalculator(fz.model, device=device)
        out = []
        for s in structures:
            at = ad.get_atoms(s)
            at.calc = calc
            out.append(at.get_potential_energy())
        return np.array(out), "orb_models ORBCalculator(model).get_potential_energy"
    if b == "sevenn":
        from sevenn.calculator import SevenNetCalculator
        calc = SevenNetCalculator("7net-0", device=device)
        out = []
        for s in structures:
            at = ad.get_atoms(s)
            at.calc = calc
            out.append(at.get_potential_energy())
        return np.array(out), "sevenn SevenNetCalculator('7net-0').get_potential_energy"
    if b.startswith("mattersim"):
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(load_path=fz.weights, device=device)
        out = []
        for s in structures:
            at = ad.get_atoms(s)
            at.calc = calc
            out.append(at.get_potential_energy())
        return np.array(out), "mattersim MatterSimCalculator(load_path=<same .pth>).get_potential_energy"
    raise ValueError(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(E.BACKBONE_INFO))
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mids = [ln.strip() for ln in open(a.ids) if ln.strip()][:a.n]
    structures = E.load_structures(mids, "initial")
    fz = E.build_featurizer(a.backbone, a.device)
    fz.want_energy = True
    t0 = time.time()
    feats, failures, _, e_ext = E.featurize_robust(fz, structures, 32, mids)
    t_ext = time.time() - t0
    t0 = time.time()
    e_off, how = official_energies(a.backbone, fz, structures, a.device)
    t_off = time.time() - t0
    n_at = np.array([len(s) for s in structures])
    d = np.abs(e_ext - e_off)
    res = {"backbone": a.backbone, "n": len(mids), "failed": len(failures), "official_path": how,
           "extract_energy_source": E.ENERGY_INFO[a.backbone], "weights_sha256": E.sha256_file(fz.weights),
           "max_abs_diff_eV": float(d.max()), "max_abs_diff_eV_per_atom": float((d / n_at).max()),
           "mean_abs_diff_eV_per_atom": float((d / n_at).mean()),
           "seconds_extract": t_ext, "seconds_official": t_off,
           "material_id": mids, "e_extract": e_ext.tolist(), "e_official": e_off.tolist(), "n_atoms": n_at.tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[check_energy_pass] {a.backbone}: n={len(mids)} max|dE|={res['max_abs_diff_eV']:.3g} eV "
          f"({res['max_abs_diff_eV_per_atom']:.3g} eV/atom) official={how}", flush=True)


if __name__ == "__main__":
    main()
