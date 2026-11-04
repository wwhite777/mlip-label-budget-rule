# Reproduction routes

## Restore, check and replay the reported evidence

The archive restore command in the README reconstructs exact files and verifies each SHA-256. Move the restored `cache`, `data` and `result` directories into the repository root. The repository's public protocols and the archive's relocated stage manifests use those paths. Use the pinned light CPU environment and thread settings from the README.

```bash
"$PYTHON" tools/check_protocol.py
"$PYTHON" tools/check_stages.py
"$PYTHON" tools/replay_readings.py --out-root runs/replayed
"$PYTHON" tools/replay_mechanism.py --out-root runs/mechanism_replayed
"$PYTHON" reporting/readme_tables.py
"$PYTHON" reporting/readme_tables.py --check
```

Inference replay recomputes every primary and secondary reading in F1–F6 from the reported point estimates, per-draw AP values and paired bootstrap arrays, then checks exact agreement with the archived readings and both judges. It does not refit models or resample labels. Mechanism replay recomputes four descriptive tables from held-out decoder predictions and the archived prepared targets. It does not refit decoders. Neither replay route needs external model packages, publisher label files or checkpoints.

The archive also holds the per-unit held-out scores, label draw IDs, inner-split IDs, fine-tuning test-energy tables and all inference inputs. These support rebuilding the inference from predictions and inspecting data separation. The prepared mechanism arrays contain the scientific targets used in that descriptive analysis; they are derived WBM data under the stated attribution terms.

## Refit original CPU stages

Obtain the byte-exact publisher summary and energy basis using `tools/download_inputs.py`. Its source URLs and hashes are in `data_sources.json`. Restore the accepted features, references and split manifests from the archive. Features and reference tables are checked before fitting; only the guarded label reader opens the outcome columns.

```bash
"$PYTHON" tools/check_inputs.py --kind cpu
"$PYTHON" pipeline/integrity.py
"$PYTHON" -m confirm_runner.runner --stage CPU-P --stage-root runs/stages --out-root runs/stages/CPU-P --workers 7
"$PYTHON" -m confirm_runner.runner --stage CPU-S --stage-root runs/stages --out-root runs/stages/CPU-S --workers 7
```

CPU-P writes its stage manifest and the primary judge manifest. CPU-S verifies both before proceeding. Use a fresh `runs/stages` directory for new numerical runs. Wall times and source paths in new output metadata differ from the reported archive; compare scientific arrays and readings separately.

## Fine-tuning and mechanism

The reported fine-tuning stages are complete. Replaying their F4 inference uses the first route above and requires no weights. Repeating training requires the publisher's initial and relaxed structures, elemental references and the exact MACE and ORB checkpoint versions listed in `DATA_SOURCES.md`. Use the model analysis environment for the launcher and numerical light environment for evaluation. The launcher runs one unit at a time and keeps the fixed full-model recipe, train/test IDs and completion checks.

```bash
export PYTHON="$PWD/.venv-analysis/bin/python"
bash pipeline/confirm_ft_launcher.sh 1 --stage FT-P --stage-root "$PWD/runs/stages" --out-root "$PWD/runs/stages/FT-P/energies"
export PYTHON="$PWD/.venv-cpu/bin/python"
"$PYTHON" -m confirm_runner.runner --stage FT-P --stage-root runs/stages --out-root runs/stages/FT-P --ft-root runs/stages/FT-P/energies --workers 7
export PYTHON="$PWD/.venv-analysis/bin/python"
bash pipeline/confirm_ft_launcher.sh 1 --stage FT-S --stage-root "$PWD/runs/stages" --out-root "$PWD/runs/stages/FT-S/energies"
export PYTHON="$PWD/.venv-cpu/bin/python"
"$PYTHON" -m confirm_runner.runner --stage FT-S --stage-root runs/stages --out-root runs/stages/FT-S --ft-root runs/stages/FT-S/energies --workers 7
"$PYTHON" -m confirm_runner.runner --stage mechanism --stage-root runs/stages --out-root runs/stages/mechanism --workers 1
```

The training commands use the positional device index accepted by the preserved launcher. They are optional GPU recipes; the CPU replay and all release checks run without a GPU. For an eight-structure CPU training smoke, after obtaining the pinned publisher inputs and weights:

```bash
export PYTHON="$PWD/.venv-analysis/bin/python"
"$PYTHON" tools/check_inputs.py --kind ft-smoke
bash pipeline/confirm_ft_launcher.sh 1 --dry-run --out-root "$PWD/runs/ft_smoke"
bash pipeline/confirm_ft_launcher.sh 1 --dry-run --out-root "$PWD/runs/ft_smoke"
```

The second invocation verifies and skips completed smoke units. These smoke units cannot serve as real fine-tuning results. The release checks do not repeat training. The public archive excludes completion markers and training runtime records along with the weights, so archived energy tables are used for scientific replay; new training uses its own fresh output directory.

Mechanism fitting uses the original probe functions on all outer-training rows. It is a descriptive analysis beyond the label-budget learner setting. Its adapter sets top-5-percent selection to the specified ceiling rule. One worker uses four native threads. The historical terminal fine-tuning cap path is preserved in the code, but no incomplete cell is converted to a reading; this release reports complete fine-tuning stages.

## Refit the relaxed-reference extension

Restore all ten relaxed-reference CSV/provenance files from the archive as well as the original inputs. The extension verifies the specification, code identities and the entire six-stage predecessor chain before inputs or labels are read. Its missing-value policy passes NaN to histogram gradient boosting, ranks missing label-free predictions last and uses a strict five-source mean for the relaxed ensemble. It never drops rows or imputes values. Infinite inputs stop execution.

```bash
export PYTHON="$PWD/.venv-cpu/bin/python"
"$PYTHON" -m confirm_runner.relaxed --stage CPU-R --stage-root data/stages --out-root runs/extension/CPU-R --workers 7
```

This uses the reported predecessor evidence. After regenerating the entire original chain instead, supply `--stage-root runs/stages`. The extension destination must remain outside the predecessor directory. CPU-R preserves the five fixed source heads, label draws, inner splits, paired chemical-system bootstrap and seed restart in each design. A5fRS is descriptive and excluded from F5, F6 and the extension rule. Its separate judge never revises the original rule.

## Feature and single-point reference regeneration

These commands require the separately pinned model environments and downloaded checkpoint bytes. They are not needed to reproduce reported inference. The archive's accepted feature arrays remain the inputs to the reported study; regenerated arrays must be compared and versioned rather than silently substituted. An earlier equivalence check failed for the accepted regenerated features, and training-set overlap remains unverified.

```bash
mkdir -p cache/features cache/reference
.venv-analysis/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
ids = pd.read_csv('cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz', usecols=['material_id'], keep_default_na=False).material_id
Path('cache/features/wbm_all_ids.txt').write_text('\n'.join(ids) + '\n')
PY
for backbone in mace_medium chgnet orb; do
  .venv-analysis/bin/python src/featurize/extract.py --backbone "$backbone" --structures initial --ids cache/features/wbm_all_ids.txt --out "cache/features/wbm_full_${backbone}_initial.npz" --energy --energy-out "cache/reference/sp_energy_${backbone}_initial.npz" --device cpu --batch 32
done
for backbone in sevenn mattersim_5m; do
  .venv-e3nn/bin/python src/featurize/extract.py --backbone "$backbone" --structures initial --ids cache/features/wbm_all_ids.txt --out "cache/features/wbm_full_${backbone}_initial.npz" --energy --energy-out "cache/reference/sp_energy_${backbone}_initial.npz" --device cpu --batch 32
done
for backbone in mace_medium sevenn mattersim_5m orb; do
  .venv-analysis/bin/python src/reference/zero_label.py sp --backbone "$backbone" --refs mbd1.0.0 --n-workers 1 --corrections cache/reference/mp2020_corr_pmg2023.5.10_initial.npz --corrections-main cache/reference/mp2020_corr_pmg2025.10.7_initial.npz
done
.venv-analysis/bin/python src/reference/zero_label.py sp --backbone chgnet --refs mbd1.0.0 --no-mp2020 --n-workers 1
```

CHGNet receives no second MP2020 correction. Relaxed references are published relaxation predictions, identified by source and output hashes in the ten archived reference files. The extension assumes, without proving, their checkpoint identity with the frozen encoders. Its released tables preserve missing rows and the prescribed rank policy.

## Hashes, transformation boundaries and numerical comparison

`ORIGINAL_HASHES.json` records exact original protocol, contract, receipt and stage identities. Public protocol views keep the scientific settings and remove private operational metadata. Archive file bindings distinguish original bytes from public bytes when string metadata was transformed. Numerical members of transformed NPZ files are copied byte-for-byte. Local stage manifests are rebuilt for the public paths and exported scientific files, with predecessor hashes linked in order. They do not reproduce the original stage-manifest hashes.

`tools/check_protocol.py` checks the original scientific field inventory, the extension field inventory and the unchanged original rule in the extension document. The integrity gate binds all source files before real access. `tools/check_stages.py` verifies all exported outputs. `tools/compare_scientific.py` compares numerical arrays between roots. Exact reruns can depend on the pinned library versions and hardware; do not turn a changed numerical run into the reported result by updating its hashes alone.
