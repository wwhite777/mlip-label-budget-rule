# Data sources and licences

The pinned WBM structures and summary, MP energy basis and elemental reference entries are distributed through [Matbench Discovery data files](https://doi.org/10.6084/m9.figshare.22715158) under CC BY 4.0. Credit the original WBM data and Matbench Discovery when reusing them. The exact byte variants below are required; the newer JSON-lines encodings have different hashes. Raw files are downloaded from the publisher rather than redistributed here.

| Input | Publisher file | Licence |
|---|---|---|
| summary | [Figshare 64706751](https://ndownloader.figshare.com/files/64706751) | CC BY 4.0 |
| initial_structures | [Figshare 40344466](https://ndownloader.figshare.com/files/40344466) | CC BY 4.0 |
| relaxed_structures_ft_targets | [Figshare 40344463](https://ndownloader.figshare.com/files/40344463) | CC BY 4.0 |
| element_basis | [Figshare 52080797](https://ndownloader.figshare.com/files/52080797) | CC BY 4.0 |
| ft_elemental_references | [Figshare 40387775](https://ndownloader.figshare.com/files/40387775) | CC BY 4.0 |

[WBM data: Wang, Botti and Marques](https://doi.org/10.1016/j.commatsci.2020.110064). [Matbench Discovery](https://github.com/janosh/matbench-discovery). Exact paths and SHA-256 are in `data_sources.json`. The structure entries supply drawn FT training targets; evaluation uses initial structures and the fixed MP hull.

| Checkpoint | Recorded licence | Source |
|---|---|---|
| mace_medium | [MIT](https://github.com/ACEsuit/mace-foundations/blob/1ff8786eb/LICENSE) | [Weights](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model) |
| chgnet | [BSD-3-Clause](https://github.com/CederGroupHub/chgnet/blob/d55f185199f/LICENSE) | [Weights](https://github.com/CederGroupHub/chgnet/raw/refs/heads/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) |
| sevenn | [GPL-3.0](https://github.com/MDIL-SNU/SevenNet/blob/8ce2c9d4/LICENSE) | [Weights](https://raw.githubusercontent.com/MDIL-SNU/SevenNet/dff008ac9c53d3/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth) |
| mattersim_5m | [MIT](https://github.com/microsoft/mattersim/blob/d0a52e64fc/MODEL_CARD.md) | [Weights](https://github.com/microsoft/mattersim/raw/refs/heads/main/pretrained_models/mattersim-v1.0.0-5M.pth) |
| orb | [Apache-2.0](https://github.com/orbital-materials/orb-models/blob/3e9580809/LICENSE) | [Weights](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt) |

The code imports the developers’ installed packages; it does not vendor their implementation or weights. Keep each upstream licence and attribution with any separately downloaded artifact. These licence identifiers come from the locally retained catalogue snapshot, rather than a live recheck.

Derived study scores, metrics, ledgers, tables and figures are CC BY 4.0. Attribute Woncheol Jeong and Hayoung Oh and the upstream WBM/Matbench Discovery data. The software itself is MIT.

The accompanying archive inventory excludes raw WBM/MP downloads and all model weights. It includes study-derived splits, feature/reference arrays and stage outputs, with upstream attribution retained. Full protocol and feature regeneration routes are in `REPRODUCTION.md`.

## Relaxed references and version 2 payload

The extension uses published Matbench Discovery relaxation predictions (IS2RE: initial structure to relaxed energy) for MACE-MP-0, CHGNet 0.3.0, SevenNet-0, MatterSim v1.0.0-5M and ORB v3. The archive includes the derived `result/reference/zero_label_is2re_*.csv.gz` tables and their adjacent JSON provenance files. Those records identify each upstream prediction file by SHA-256 and MD5, record missing-row coverage, and bind the derived table's SHA-256. Their source family is the same WBM/Matbench Discovery distribution above, recorded as CC BY 4.0. No new online licence check was performed for this offline candidate.

Frozen feature matrices and the mechanism's prepared feature/target matrices are numerical outputs derived from the attributed WBM structures and reference data. They contain no model implementation or checkpoint weights. They are distributed with the study-derived numeric evidence under CC BY 4.0; the upstream model software licences are retained in the table above and are not replaced by the repository's MIT licence. The GPL-3.0 SevenNet implementation itself is not vendored or relicensed.

The version 2 archive also includes the original and relaxed-reference stage predictions, exact split and draw IDs, per-unit numerical fit summaries, fine-tuning test/epoch-zero energies, mechanism decoder predictions, bootstrap samples and inference records. Publisher structure archives, raw model weights, training runtime records and local environments are excluded. Checkpoint identity between the published relaxed predictions and the frozen encoder features is assumed rather than established.
