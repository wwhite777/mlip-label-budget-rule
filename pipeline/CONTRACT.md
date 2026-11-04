# Scientific contract — v2.3.1


Issued `2026-10-22`. Twin: `protocol.yaml`.
All arms were developed on PILOT. A1 was seen post hoc on CONFIRM: DESCRIPTIVE everywhere.


## 1. Primary claim

For the five frozen draws, conditional on fitted models, the exposure-informed claim is: at the tested label budgets of 300 and 1,000, the frozen correction set of F2 does not exceed the strongest measured zero-shot comparator by the practical margin; at 10,000, at least one A5f tree head on frozen features plus zero-shot hull distances adds beyond that margin. The 3,000-label result and mechanism are reported separately. A1-equivalent residual heads nested in A5f inputs were already seen on CONFIRM.

Five specified checkpoints and tested discrete budgets only; no continuous threshold or below/above-budget claim. FT has F4; A1 is descriptive. F1/F2: confirmatory, exposure-informed; not independent confirmations of an unseen population.


Scope: listed population/checkpoints/designs and discrete budgets only; success rule below, no continuous threshold.

## 2. Population, labels, inputs

Unique-prototype universe `215488`; CONFIRM `172689`
structures / `59319` systems; PILOT `42799` /
`14830`. Reuse label-blind partition seed `20261002` and
existing manifests. Source Figshare IDs: summary `64706751`, initial structures
`40344466`, relaxed FT targets `40344463`.
Stable class `1` iff `e_above_hull_mp2020_corrected_ppd_mp` ≤
`0.0` eV/atom: fixed-MP-hull stability. Formation label:
`e_form_per_atom_mp2020_corrected`. Never regenerate labels. PBE gap threshold `1.0`
eV identifies only the carried-over v1.2 confirmatory record; no new gap analysis.

Primary inputs: initial structures, frozen mean-pooled scalar-node features. MP2020 once on initial structures via
pymatgen `2023.5.10`; CHGNet only `_initial_nomp2020`, no second correction. Fixed MP composition hull, never rebuilt
from WBM labels or corrected-arm energies. Raw fractions use the pinned `89`-element basis,
atomic-number order. The accepted regenerated feature arrays failed two equivalence-check points, including the retained numerical-equivalence check. The explicit feature_inputs hashes define the accepted inputs; legacy registry hashes differ. No silent replacement or bypass of these checks is allowed.

K = `5`, ordered: `MACE-MP-0 medium (2023-12-03 L1)`; `CHGNet 0.3.0`; `SevenNet-0 (11Jul2024)`;
`MatterSim v1.0.0 5M`; `ORB v3 conservative-inf-mpa`. ORB has non-invariant descriptors and broader pretraining; all listed checkpoints enter the families.

## 3. Designs, draws, selection and arms

P, primary: `5`-fold chemical-system holdout, v1.2 manifest, assignment seed
`0`. All designs use replicate IDs `[0,1,2,3,4]`. In P, canonical outer-training IDs;
choose exactly n without replacement with default_rng(a*f+r+c), a=`1000`,
c=`0`. Share draws across arms. Budget means n per outer-fold learner, not total labels
over folds/replicates; same seed formula across budgets, no enforced nesting.

S-a/S-b, secondary: pilot manifests; train rounds `[1,2]`, test rounds
`[3,4,5]`. Shared pool `80311` structures /
`33645` systems. S-a test `92378` /
`42998`, shared train/test systems
`17324`. S-b excludes all training-pool systems, removing
`57226` structures; test `35152` /
`25674`, shared systems `0`.
S-a permits system overlap. The `campaign_draw` sampler: permute sorted pool systems using seed a+b*n+r,
a=`500000`, b=`1000`; concatenate canonical IDs within
systems, truncate the last to exact n. This is not uniform structure sampling. Fit once for both variants; S-b
restricts S-a scores. Inner fold index `0`; no test-driven retuning.

P budgets; S omits the descriptive budget. Confirmatory n: `[300,1000,3000,10000]`; descriptive only: `[100]`; FT only its family budgets.
Inner validation: permute sorted draw systems with default_rng(a*f+r+c), a as P, c=`7`; take
complete systems until cumulative count first reaches `0.2`*n. For inner-selected CPU arms, select highest inner AP for each
learner, first grid on exact ties; refit all n. Train-only scaling; no outside-draw label in fitting, selection or priors.
Preserve the CPU implementation’s single-class fallback (YAML `inner.single_class`): first grid, validation-prevalence diagnostic;
full single-class draw gives constant scores, otherwise refit first grid. C1 tunes AP with first-grid ties. No redraw;
empty split hard error (C10).

F_k=frozen features, c=fractions, ĥ_k=stored zero-shot hull distance, f=DFT formation label. Exact CPU/campaign arms:

| Arm | Definition |
|---|---|
| R0_k | score −ĥ_k, label free |
| R0* | max of R0_k APs on the same evaluation set, recomputed each bootstrap; strongest measured zero-shot comparator; evaluation-label oracle, not a deployable selection rule |
| ENS (code E0) | negative mean of checkpoint hull distances; distinct from FT atomic E0 offsets |
| RR (code C1_k) | ridge of f−predicted formation on raw c; score −(ĥ_k+b+cδ), fixed MP hull, unpenalized intercept retained across folds |
| A4 | HistGBM on [all ĥ,c]; the tree code includes composition; logistic/best menu variants descriptive |
| A5f_k | HistGBM on [float64(F_k),all ĥ], stable-class probability; fixed PILOT modal point, no inner selection; fit all n labels |
| A1_k | [float64(F_k),own ĥ_k], logistic/HistGBM/inner-selected best; DESCRIPTIVE everywhere |
| A5_k | Inner-selected arms HistGBM; DESCRIPTIVE everywhere |
| FT_k | MACE/ORB, FT-full, frozen foundation atomic E0; active F4 |

| CPU grid | Values |
|---|---|
| HistGBM rates / leaves | `[0.05,0.1]` / `[15,31]` |
| Iterations / seed | `300` / `0`; no early stopping or scaling |
| Descriptive logistic C / iterations | `[0.01,0.1,1.0,10.0]` / `2000`; unweighted, train-only StandardScaler, lbfgs |
| C1 alpha order | `[0.01,0.1,1.0,10.0,100.0]`; float64 SVD, raw fractions, intercept |

Grid order: learning rate then leaves; other defaults pinned to sklearn `1.7.2`.
Across types, logistic wins exact ties. Confirmatory A4 uses its separately tuned HistGBM; A5f uses its PILOT-fixed HistGBM, never a menu winner.

### FT variant: G004 (active)

G004 was chosen on PILOT (development) among G002–G004 (G001 void); none exceeded zero-shot in the pilot comparison.
FT-full uses relaxed training geometries and raw CSE totals. Foundation atomic E0 stays frozen and excluded from AdamW;
epoch zero is the foundation model and is selectable by strict inner-validation per-atom MSE improvement. Fresh
foundation refit uses all n draw labels for exactly the selected epoch count. No test targets enter optimization or
energy conversion. Scoring uses initial geometry, MP2020 once, rounded elemental references and the pinned fixed MP hull.
Training seeds: P uses multiplier times outer fold plus draw; S uses multiplier times its reserved fold plus draw,
as recorded in the operational values below. The draw and inner split still follow the shared samplers.
No scheduler/replay; float32-highest. Untouched-checkpoint identity tolerance and optimizer constants are recorded below.
Descriptive FT at n equal to the high family budget: P, draw zero only; no F4 reading for that addition.

## 4. Scores, metrics, aggregation

For each r in P, pool its held-out-fold predictions over CONFIRM; M_r=step-wise structure-weighted
`average_precision_score`, M=mean_r(M_r). Never average fold AP or replicate scores. S computes AP per replicate on
each fixed variant test set, then averages AP. References are deterministic. Report counts, positives, prevalence;
retain IDs/scores, splits/draws, selected parameters, flags, versions, hashes and costs.

Descriptive AUROC and precision: top fraction `0.05`; P uses ceil(fraction*N_f) per fold,
pools hits/count, then averages replicates; S uses ceil(fraction*N_test) per variant, then averages replicates.
Ties: score descending, material_id ascending. The mechanism probe uses ceil rounding. The legacy rounding quantity q=`0.5` is inactive.
 Nonfinite labels hard error; complete finite feature/reference
coverage required, no silent intersection/imputation. Partial folds/replicates receive no reading (C11).

## 5. Hypotheses, families, readings, success

Paired chemical-system cluster bootstrap B=`2000`, new seed `20261022`; conditional on fitted
models. Sort evaluation systems, default_rng(seed), resample as many systems as present with replacement; carry all
structures and method/replicate scores together. Restart RNG per design. P resamples CONFIRM systems; S test systems
only. No refitting. Recompute weighted M_r then M, R0*, and F2's maximum inside every draw; maxima follow replicate
aggregation, never winning-arm switching per replicate/fold.

Unadjusted two-sided nominal percentile coverage `0.9`, percentiles `[5,95]`, numpy
linear interpolation. p↑=`(1 + #{d* <= m}) / (B + 1)` for `d <= m`;
p↓=`(1 + #{d* >= m}) / (B + 1)` for `d >= m`. Separate directional Holm step-down at
alpha=`0.05` within each family AND design; Holm ties by contrast name. No global error-control claim.
Intervals do not determine readings; nonfinite bootstrap contrasts prevent readings. Undefined descriptive AUROC
draws are omitted from intervals with counts; this omission rule is new in v2.3, not inherited from v1.2.

`Oracle and contender-maximum percentile bands are descriptive, nominal percentile summaries; their coverage can be poor at ties and they must not be presented as calibrated 90 % confidence intervals.`

Common m=`0.02` absolute AP, unchanged rationale: `below about 2 points of average precision a practitioner would not change pipelines given the cost of 3,000 DFT labels`.
The common practical threshold is retained without fitting budget-specific margins to PILOT.
At n=`300`, a minimal label purchase still needs a pipeline-changing gain.
At n=`1000`, the same practical floor prevents a smaller significance-driven threshold.
At n=`3000`, the historical label-cost rationale applies directly.
At n=`10000`, the larger investment must clear at least the same practical floor.
Low-budget DOES_NOT_EXCEED was expected from PILOT for related controls and carries limited new information; F1 at its high budget is the informative test.
ADDS: Holm-adjusted p↑≤alpha;
DOES_NOT_EXCEED: adjusted p↓≤alpha, improvement below +m, not zero benefit; otherwise INCONCLUSIVE. Opposite readings
cannot both hold. P primary, S-a/S-b repeat the same family structure separately as secondary analyses:

| Family | Contrasts |
|---|---|
| F1 | n=`10000`: M(A5f_k)−M(R0*) and M(A4 HistGBM)−M(R0*); `6` contrasts, IUT ↑ and fixed-ORB ↓, separate family Holm |
| F2 | n∈`[300,1000,3000]`: d_max(n)=max over {A5f_k,A4 HistGBM,C1_k,E0 mean} of M−M(R0*); `12` contenders, `3` contrasts, ↓ Holm against fixed ORB, ↑ descriptive only; contender maximum recomputed after draw aggregation |
| F3 | n=`10000`: M(C1_k)−M(R0_k); `5` contrasts, separate ↓/↑ Holm; negative control at positive m |
| F4 | n∈`[1000,3000]`, MACE/ORB: M(FT_k)−M(R0_k); `4` planned contrasts per design; active; separate ↑/↓ Holm per design |

ADDS (F1): Exceeding the strongest measured zero-shot comparator by m requires exceeding every fixed reference by m, so the null is a union of component nulls. Compute the five paired one-sided component p-values, set p_IUT = max_k p↑_k, apply the family Holm adjustment to those per-contrast IUT p-values, and read ADDS only when adjusted p_IUT ≤ alpha.

DOES_NOT_EXCEED (F1/F2): ORB v3 is fixed before new CONFIRM evaluation as the best PILOT design-P reference, and M(arm) − M(R0*) ≤ M(arm) − M(R0_ORB). A family-Holm rejection of the down null against this fixed reference is therefore a sufficient condition for the oracle negative reading; F2 retains its contender maximum, which is no more liberal than any null contender's scalar test with a fixed reference.

Every DOES_NOT_EXCEED reading is reported with its down_estimate and down_interval against fixed ORB; the oracle estimate is labelled descriptive.

F2 up: Maximizing over contenders can select a favorable bootstrap arm and does not supply valid confirmatory ADDS inference. Any F2 up result is labelled DESCRIPTIVE_ADDS, including the separately reported budget, and cannot affect success.

Report sample SD of M_r next to **every reading**, plus the reference and contrast-draw SDs; for maxima use the point-estimate winning arm held fixed across draws. Interpret results “for the five frozen draws, conditional on fitted models”; cluster intervals do not include retraining, new label draws or selection variability.

F1 tests the combined pipeline against individual zero-shot references; **F1 does not isolate an encoder-feature contribution beyond A4/ENS**.

Success exactly: in P, F2 DOES_NOT_EXCEED at BOTH `[300,1000]` AND F1 A5f ADDS for at least
`1` A5f contrast; A4 does not count. Remaining F2 budget, F3, F4 and S results do not change this rule.
Judge success as soon as complete design-P F1/F2 rows exist, even if F4 is missing or the mechanism fails. F3, F4, S and mechanism completion are reported separately. Failed or unavailable ancillary CPU-P units are explicitly recorded; no partial F3 family is read and its absence cannot prevent a complete F1/F2 judgment. Required F1/F2 unit/population failures still stop that judgment.

Anything else is reported as observed; no outcome-dependent change to the specified scope.

## 6. PILOT development and descriptive analyses

PILOT development exposure and the failed numerical-equivalence check remain disclosed.

C14: no A2/A3 rerun; cite the delta record descriptively. Every inner-selected A5 is descriptive. Every emitted cell outside exact families is descriptive, no readings: every A1; other n/design cells; AUROC;
top-fraction precision; per-round S test results; mechanism; numerical noise. A2/A3 and menu/median variants, if
emitted, are descriptive; this catch-all does not authorize extra runs. Band gap is only the old confirmatory record.
Descriptive AP/AUROC use paired conditional bootstrap, no Holm/readings. Precision/strata/noise follow source reporting.

Mechanism: probe rerun under P, all outer-training CONFIRM rows, no n-label budget or replicate dimension, no readings.
Inputs features / composition / concatenation; independent targets e_k=own formation prediction, h=e_k−ĥ_k,
f=DFT formation, y=DFT hull distance, error=f−e_k. Ridge alphas `[0.01,0.1,1.0,10.0,100.0,1000.0]`; train-only scaling/intercept,
leave each other outer-training fold out, choose by pooled inner MAE per target, first-alpha ties, refit all training.
Singleton HistGBM: `{"learning_rate":0.1,"max_leaf_nodes":31,"max_iter":200,"min_samples_leaf":20,"l2_regularization":1.0,"early_stopping":false,"random_state":0}`. Report MAE/RMSE/bias in meV/atom, R², explained variance; pooled/fold AP and
precision for −(decoded f−exact h), −decoded y, −(decoded e_k−exact h), −(decoded e_k−decoded h).
eV→meV factor `1000`; common-h check tolerance `1e-05` eV/atom;
keep checkpoint rounding. Flip summaries: correction flags and joint strata, element count, inclusive |y| windows
`[25,50,100]` meV/atom; counts, prevalence, rates and flip shares. Finite decoders do not prove
irreversible feature information loss; full-range R² does not establish near-hull precision.

Single-unit gaps <`0.01` AP are not interpreted. arms local diagnostic:
relative Gaussian scale `1e-07`, seeds `[0,1,2,3,4]`, fixed selected configurations,
n=`3000`, f=`0`, r=`0`; max absolute inner-AP change.
Preserve input dtype; perturb specified arm inputs, scoring/target references fixed, no fraction renormalization.
Apply the fixed PILOT pooled stability check below before interpreting F1/F2. No CONFIRM tuning; failed gate (ii) disclosed.

## 7. Cost and caps

One G004 variant per FT checkpoint; shared-S fitting, distinct S-a/S-b evaluation. Full F4 plus the high-budget
P descriptive addition are estimated from actual G004 phase timings, separating training-scaled time from test scoring.
CONFIRM tests exceed PILOT tests; the estimate and scoring share are recorded in the numeric values below and
the `cost` section of `protocol.yaml`. Early stopping can change substantially; the planning range is not a bound. Count allocated time,
failed runs/retries and scoring once per driver. Epoch-zero/final FT rows share cost. Stop at the cumulative cap;
report incomplete cells, never silently shrink scope. Speed claims require measurements.

## 8. Mandatory baselines

All listed baselines mandatory, with shared populations/draws; named HistGBM rows govern.

## Prior evaluation exposure

The protocol was fixed before the evaluation population was scored for these methods; earlier related results on the same population are listed in the exposure table.

A1-equivalent residual heads, feature-only heads, budget sweeps, delta and mechanism controls had prior results. ORB had results at every scheduled budget. Earlier low-budget deficits and high-budget residual-head gains informed F1/F2; these are exposure-informed comparisons. Exact combined A5f, A4, ENS, RR and full fine-tuning recipes had no recorded prior outcomes. The table in `EXPOSURE_TABLE.csv` distinguishes retained outcomes from reconstructed scope.

Training-set membership and WBM overlap remain unverified for all five checkpoints. Accepted regenerated feature arrays retain an earlier failed equivalence check; their explicit hashes define the reported inputs.

The full typed scientific settings, including resource estimates, FT recipe, numerical stability condition, and all inference settings, are given in `protocol.yaml`.

## Numerical calibration and stability

The five fixed A5f sources pass the PILOT pooled stability criterion: the maximum absolute AP shift across seeds 0–2 under relative Gaussian noise of 1e-7 is at most 0.005. A missing or failed source makes all F1/F2 readings descriptive, preserving every source and the family sizes. The three-source numerical results are in `stability_check_3sources.csv`; the fixed verdict map includes all five sources.

Exact binary-construction checks give scalar tail probabilities 0.0563 at N = 1,000 and 0.0510 at N = 59,319. A separate unequal-cluster stress construction gave error 0.370. The reproduced calibration fixtures check regression values with Monte Carlo tolerance; they do not establish general family error control or a finite-sample guarantee. The fixed approximation uses no size correction. Oracle and contender-maximum percentile bands remain nominal descriptive summaries with potentially poor coverage at ties.

## Typed operational values

The table is generated from the YAML and checked for exact typed-value concordance. It includes every numerical or boolean scientific setting, including the complete FT optimizer, seeds, noise settings, fit counts, measured timings, resource estimates and caps. Scientific campaign round identifiers are population definitions.

| Field path | Value |
|---|---|
| `population.universe_n` | `215488` |
| `population.confirm_n` | `172689` |
| `population.confirm_systems` | `59319` |
| `population.pilot_n` | `42799` |
| `population.pilot_systems` | `14830` |
| `population.partition_seed` | `20261002` |
| `population.summary.figshare_file_id` | `64706751` |
| `population.initial_structures.figshare_file_id` | `40344466` |
| `population.relaxed_structures_ft_targets.figshare_file_id` | `40344463` |
| `labels.stability.threshold_eV_atom` | `0.0` |
| `labels.stability.positive_class` | `1` |
| `labels.regenerate` | `false` |
| `labels.gap_record.threshold_eV` | `1.0` |
| `inputs.composition_dim` | `89` |
| `backbones.count` | `5` |
| `designs.P.n_folds` | `5` |
| `designs.P.fold_seed` | `0` |
| `designs.P.draw_fold_multiplier` | `1000` |
| `designs.P.draw_seed_offset` | `0` |
| `designs.S.train_rounds.0` | `1` |
| `designs.S.train_rounds.1` | `2` |
| `designs.S.test_rounds.0` | `3` |
| `designs.S.test_rounds.1` | `4` |
| `designs.S.test_rounds.2` | `5` |
| `designs.S.train_n` | `80311` |
| `designs.S.train_systems` | `33645` |
| `designs.S.sampler_seed_offset` | `500000` |
| `designs.S.sampler_budget_multiplier` | `1000` |
| `designs.S.inner_fold_index` | `0` |
| `designs.S.S-a.test_n` | `92378` |
| `designs.S.S-a.test_systems` | `42998` |
| `designs.S.S-a.shared_train_test_systems` | `17324` |
| `designs.S.S-b.test_n` | `35152` |
| `designs.S.S-b.test_systems` | `25674` |
| `designs.S.S-b.excluded_test_n` | `57226` |
| `designs.S.S-b.shared_train_test_systems` | `0` |
| `replicates.ids.0` | `0` |
| `replicates.ids.1` | `1` |
| `replicates.ids.2` | `2` |
| `replicates.ids.3` | `3` |
| `replicates.ids.4` | `4` |
| `replicates.mean_scores_allowed` | `false` |
| `budgets.confirmatory.0` | `300` |
| `budgets.confirmatory.1` | `1000` |
| `budgets.confirmatory.2` | `3000` |
| `budgets.confirmatory.3` | `10000` |
| `budgets.descriptive_only.0` | `100` |
| `inner.fraction` | `0.2` |
| `inner.seed_offset` | `7` |
| `models.hist_gbm.learning_rate.0` | `0.05` |
| `models.hist_gbm.learning_rate.1` | `0.1` |
| `models.hist_gbm.max_leaf_nodes.0` | `15` |
| `models.hist_gbm.max_leaf_nodes.1` | `31` |
| `models.hist_gbm.max_iter` | `300` |
| `models.hist_gbm.random_state` | `0` |
| `models.hist_gbm.early_stopping` | `false` |
| `models.logistic_descriptive.C.0` | `0.01` |
| `models.logistic_descriptive.C.1` | `0.1` |
| `models.logistic_descriptive.C.2` | `1.0` |
| `models.logistic_descriptive.C.3` | `10.0` |
| `models.logistic_descriptive.max_iter` | `2000` |
| `models.c1.alphas.0` | `0.01` |
| `models.c1.alphas.1` | `0.1` |
| `models.c1.alphas.2` | `1.0` |
| `models.c1.alphas.3` | `10.0` |
| `models.c1.alphas.4` | `100.0` |
| `models.c1.fit_intercept` | `true` |
| `arms.R0_star.deployable_model` | `false` |
| `arms.FT.seed_fold_multiplier` | `1000` |
| `arms.FT.seed_S_fold` | `5` |
| `arms.A5f.fixed_params.learning_rate` | `0.05` |
| `arms.A5f.fixed_params.max_leaf_nodes` | `31` |
| `metrics.top_fraction` | `0.05` |
| `metrics.half_round` | `0.5` |
| `inference.B` | `2000` |
| `inference.seed` | `20261022` |
| `inference.coverage` | `0.9` |
| `inference.percentiles.0` | `5` |
| `inference.percentiles.1` | `95` |
| `inference.alpha` | `0.05` |
| `margin.AP` | `0.02` |
| `margin.common` | `true` |
| `families.F1.n` | `10000` |
| `families.F1.size` | `6` |
| `families.F2.budgets.0` | `300` |
| `families.F2.budgets.1` | `1000` |
| `families.F2.budgets.2` | `3000` |
| `families.F2.size` | `3` |
| `families.F2.contenders_n` | `12` |
| `families.F3.n` | `10000` |
| `families.F3.size` | `5` |
| `families.F4.budgets.0` | `1000` |
| `families.F4.budgets.1` | `3000` |
| `families.F4.size` | `4` |
| `success.low_budgets_required.0` | `300` |
| `success.low_budgets_required.1` | `1000` |
| `success.high_min_contrasts` | `1` |
| `success.conjunction` | `true` |
| `probe.ridge_alphas.0` | `0.01` |
| `probe.ridge_alphas.1` | `0.1` |
| `probe.ridge_alphas.2` | `1.0` |
| `probe.ridge_alphas.3` | `10.0` |
| `probe.ridge_alphas.4` | `100.0` |
| `probe.ridge_alphas.5` | `1000.0` |
| `probe.tree.learning_rate` | `0.1` |
| `probe.tree.max_leaf_nodes` | `31` |
| `probe.tree.max_iter` | `200` |
| `probe.tree.min_samples_leaf` | `20` |
| `probe.tree.l2_regularization` | `1.0` |
| `probe.tree.early_stopping` | `false` |
| `probe.tree.random_state` | `0` |
| `probe.boundary_meV_atom.0` | `25` |
| `probe.boundary_meV_atom.1` | `50` |
| `probe.boundary_meV_atom.2` | `100` |
| `probe.eV_to_meV` | `1000` |
| `probe.common_h_tolerance_eV_atom` | `1e-05` |
| `descriptive.no_readings` | `true` |
| `noise.single_unit_noninterpretation_AP` | `0.01` |
| `noise.relative_gaussian_scale` | `1e-07` |
| `noise.seeds.0` | `0` |
| `noise.seeds.1` | `1` |
| `noise.seeds.2` | `2` |
| `noise.seeds.3` | `3` |
| `noise.seeds.4` | `4` |
| `noise.unit.n` | `3000` |
| `noise.unit.fold` | `0` |
| `noise.unit.replicate` | `0` |
| `noise.pre_freeze_check.n` | `10000` |
| `noise.pre_freeze_check.replicate` | `0` |
| `noise.pre_freeze_check.seeds.0` | `0` |
| `noise.pre_freeze_check.seeds.1` | `1` |
| `noise.pre_freeze_check.seeds.2` | `2` |
| `noise.pre_freeze_check.relative_gaussian_scale` | `1e-07` |
| `noise.pre_freeze_check.max_pooled_AP_shift` | `0.005` |
| `cost.gpu_cap_hours` | `90` |
| `cost.gpu_used_hours_at_issue` | `16.9` |
| `cost.gpu_remaining_hours_at_issue` | `73.1` |
| `cost.cpu_arms_estimate_hours` | `3` |
| `cost.ft_grid.P_fits` | `100` |
| `cost.ft_grid.S_fits_shared_variants` | `20` |
| `cost.ft_grid.total_fits` | `120` |
| `cost.ft_grid.descriptive_P_fits` | `10` |
| `cost.ft_grid.total_with_descriptive` | `130` |
| `cost.g004_measured_gpu_hours.mace_medium` | `0.09223073767549876` |
| `cost.g004_measured_gpu_hours.orb` | `0.05559133303817362` |
| `cost.budget_scaling_n_reference` | `1000` |
| `cost.pilot_test_n` | `8560` |
| `cost.confirm_P_test_n_total` | `172689` |
| `cost.confirm_S_a_test_n` | `92378` |
| `cost.grid_nominal_gpu_hours` | `19.376810184218904` |
| `cost.descriptive_nominal_gpu_hours` | `7.133482081991401` |
| `cost.total_nominal_gpu_hours` | `26.510292266210307` |
| `cost.F4_scoring_gpu_hours` | `2.674657412560948` |
| `cost.descriptive_scoring_gpu_hours` | `0.17425176046725452` |
| `cost.full_grid_scoring_share` | `0.10746426876098182` |
| `cost.planning_factors.0` | `0.5` |
| `cost.planning_factors.1` | `2.0` |
| `cost.grid_planning_range_gpu_hours.0` | `13.255146133105153` |
| `cost.grid_planning_range_gpu_hours.1` | `53.020584532420614` |
| `cost.grid_ceiling_proxy_gpu_hours` | `120.44191365049775` |
| `run_matrix.named_cpu.fitted.2.readings` | `false` |
| `run_matrix.named_cpu.fitted.3.readings` | `false` |
| `run_matrix.named_cpu.campaign_fit_once_for_variants` | `true` |
| `run_matrix.named_cpu.budgets_by_fit_design.P.0` | `100` |
| `run_matrix.named_cpu.budgets_by_fit_design.P.1` | `300` |
| `run_matrix.named_cpu.budgets_by_fit_design.P.2` | `1000` |
| `run_matrix.named_cpu.budgets_by_fit_design.P.3` | `3000` |
| `run_matrix.named_cpu.budgets_by_fit_design.P.4` | `10000` |
| `run_matrix.named_cpu.budgets_by_fit_design.S.0` | `300` |
| `run_matrix.named_cpu.budgets_by_fit_design.S.1` | `1000` |
| `run_matrix.named_cpu.budgets_by_fit_design.S.2` | `3000` |
| `run_matrix.named_cpu.budgets_by_fit_design.S.3` | `10000` |
| `run_matrix.FT.campaign_fit_once_for_variants` | `true` |
| `run_matrix.FT.descriptive.budgets.0` | `10000` |
| `run_matrix.FT.descriptive.replicates.0` | `0` |
| `run_matrix.FT.descriptive.readings` | `false` |
| `execution_checks.synthetic.budgets.0` | `300` |
| `execution_checks.synthetic.budgets.1` | `1000` |
| `execution_checks.synthetic.replicates.0` | `0` |
| `execution_checks.synthetic.replicates.1` | `1` |
| `execution_checks.synthetic.B` | `200` |
| `execution_checks.synthetic.scores_saved_replicates.0` | `0` |
| `ft_recipe.active` | `true` |
| `ft_recipe.epochs_max` | `100` |
| `ft_recipe.patience` | `15` |
| `ft_recipe.epoch_zero` | `0` |
| `ft_recipe.batch_size` | `8` |
| `ft_recipe.full_lr` | `0.0001` |
| `ft_recipe.betas.0` | `0.9` |
| `ft_recipe.betas.1` | `0.999` |
| `ft_recipe.eps` | `1e-08` |
| `ft_recipe.weight_decay` | `0.0` |
| `ft_recipe.identity_tolerance_eV_atom` | `2e-06` |
| `ft_recipe.epoch0_selectable` | `true` |
| `execution.active_F4` | `true` |
