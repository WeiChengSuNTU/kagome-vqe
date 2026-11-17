# graph_hva error mitigation, M2 style: readout, symmetry, ZNE and twin on layer noise

Status (2026-11-17): the full M2 comparison of the legacy study
([`legacy_m0_m1_m2.md`](legacy_m0_m1_m2.md), "Error mitigation (roadmap M2)") redone for the
graph HVA on PRB 106 paper-layer noise, at fixed parameters. ZNE alone, its methods (probability
scaling, layer folding, fits) and ZNE inside gradient descent are in
[`graph_hva_mitigation.md`](graph_hva_mitigation.md); this document compares all four M2
techniques and their combinations. Like main's M2 it has, besides the sampled table (Sections
2-3), the infinite-shot limit of every stack with the cancellation and twin-damping analysis
(Section 4), a check of which stack survives stronger noise (Section 5) and full optimisations
with the stacks (Section 6). Everything was measured on this circuit; no number is taken from
main's FakeGuadalupeV2 study.

**Summary.** At the PRB 106 strength `p_e = 1e-3` (plus readout error 0.02) readout correction
followed by symmetry filtering removes the bias (+0.4 % in the infinite-shot limit) at no extra
circuits or variance. At `3e-3` and `1e-2` it leaves +1.7 % and +10.4 %; readout + symmetry +
twin (main's M2 choice) stays within 1 %, and readout + ZNE is unbiased at about twice the
spread. Symmetry + ZNE over-corrects (-10.6 % at `1e-2`), and filtering before correcting
over-corrects by -6.8 % -- both as on main. In full optimisations (gradient descent and SPSA)
no stack changes the parameters the optimiser reaches; they change the reported energy from
10 % too high to correct.

## 1. Techniques on the logical-layer model

| technique | config | what it removes | cost | notes |
|---|---|---|---|---|
| readout correction | `mitigation.readout` | readout bit flips (bias `1 - (1 - 2q)^2` on two-body terms) | 2 one-off calibration circuits x 16384 shots per estimator | tensored per-qubit inverse; correct-then-filter when combined with symmetry (as in M2) |
| symmetry filtering | `mitigation.symmetry` | shots outside the singlet sector (`n/2` ones in every basis) | none (discards shots) | valid because `graph_hva` states are total-spin singlets for every parameter value |
| ZNE | `mitigation.zne` with `noise_scaling = "probability"` or `"layer_folding"` | gate noise on the PRB 106 locations | 3 circuits per energy (scales 1, 2, 3 or 1, 3, 5) | see [`graph_hva_mitigation.md`](graph_hva_mitigation.md); does not touch readout error |
| twin rescaling | `mitigation.rescale` | a common multiplicative damping, measured on a twin with known noise-free energy | one-off twin measurement (3 bases x 16384 shots) | the twin factor is one number per run |

New for this comparison: the logical-layer model can carry a **readout error**
(`logical_noise.readout_error = q`: every measured bit flipped with probability `q`, also in the
readout calibration). PRB 106 has none; without it readout correction would have nothing to do.
Readout correction, symmetry filtering and twin rescaling now work on `logical_noise` (trajectory
engine); the per-basis estimate is the correct-then-filter ratio of `kagome_vqe.mitigation.
counts`, with its standard error from the delta method (it matches the observed spread).

## 2. The M2 table

`configs/mitigation/graph_hva_m2_fixed.toml`, `results/20261116T181808Z-graph_hva_m2_fixed`
(data: [`data/graph_hva_m2_fixed_points.csv`](data/graph_hva_m2_fixed_points.csv)). PRB 105
2x6 rectangle, `graph_hva` `p = 2`, at the noise-free L-BFGS-B optimum (seed 1, noise-free
energy -19.656) and at a generic point (all parameters 0.3, -15.145). Noise models as M2's
columns: **gates** = depolarising `p_e = 1e-3` on the PRB 106 locations; **readout** = readout
error `q = 0.02` only; **full** = both. 1024 shots per basis; 16 repeats, each with a fresh
estimator (its own seed, calibration and twin). Entries: bias of the mean in % of the noise-free
energy +- standard error of the mean.

| stack | gates, opt | readout, opt | full, opt | gates, 0.3 | readout, 0.3 | full, 0.3 | spread (full, opt) | circuits per energy | one-off shots |
|---|---|---|---|---|---|---|---|---|---|
| none | +2.96 ± 0.12 | +7.75 ± 0.17 | +10.58 ± 0.14 | +2.72 ± 0.25 | +7.65 ± 0.26 | +10.40 ± 0.23 | 0.113 | 3 | 0 |
| readout | +2.96 ± 0.12 | -0.08 ± 0.19 | +2.99 ± 0.15 | +2.72 ± 0.25 | -0.18 ± 0.28 | +2.80 ± 0.24 | 0.121 | 3 | 32,768 |
| symmetry | +0.40 ± 0.12 | +0.55 ± 0.19 | +1.48 ± 0.12 | +0.23 ± 0.24 | +0.37 ± 0.27 | +1.41 ± 0.26 | 0.098 | 3 | 0 |
| **readout+symmetry** | **+0.40 ± 0.12** | **-0.19 ± 0.20** | **+0.27 ± 0.12** | **+0.23 ± 0.24** | **-0.29 ± 0.27** | **+0.33 ± 0.27** | **0.096** | 3 | 32,768 |
| ZNE (prob. 1,2,3, exp.) | -0.31 ± 0.28 | +8.06 ± 0.15 | +7.59 ± 0.35 | -0.36 ± 0.32 | +8.08 ± 0.38 | +7.44 ± 0.39 | 0.273 | 9 | 0 |
| ZNE (folding 1,3,5, exp.) | -0.23 ± 0.19 | +8.03 ± 0.16 | +7.59 ± 0.21 | -0.46 ± 0.24 | +8.26 ± 0.27 | +7.44 ± 0.22 | 0.165 | 9 (circuits 1x, 3x, 5x deep) | 0 |
| readout+ZNE (prob.) | -0.31 ± 0.28 | +0.21 ± 0.17 | -0.30 ± 0.38 | -0.36 ± 0.32 | +0.25 ± 0.42 | -0.44 ± 0.41 | 0.296 | 9 | 98,304 |
| readout+symmetry+ZNE (prob.) | -0.43 ± 0.25 | +0.20 ± 0.19 | -0.77 ± 0.33 | -0.48 ± 0.38 | +0.01 ± 0.42 | -0.44 ± 0.54 | 0.259 | 9 | 98,304 |
| twin | -0.47 ± 0.23 | -0.24 ± 0.23 | -0.66 ± 0.32 | -0.56 ± 0.24 | -0.06 ± 0.27 | -0.58 ± 0.34 | 0.248 | 3 | 49,152 |
| readout+twin | -0.47 ± 0.23 | -0.26 ± 0.22 | -0.67 ± 0.31 | -0.56 ± 0.24 | -0.07 ± 0.27 | -0.60 ± 0.34 | 0.247 | 3 | 81,920 |
| symmetry+twin | -0.37 ± 0.20 | +0.08 ± 0.23 | +0.18 ± 0.25 | -0.52 ± 0.25 | +0.22 ± 0.29 | -0.32 ± 0.38 | 0.200 | 3 | 49,152 |
| readout+symmetry+twin | -0.37 ± 0.20 | -0.13 ± 0.23 | -0.15 ± 0.25 | -0.52 ± 0.25 | +0.07 ± 0.30 | -0.63 ± 0.39 | 0.198 | 3 | 81,920 |

(No mean lies below the exact ground energy -19.910. Readout correction changes nothing in the
gates column, which has no readout error to correct. Wall time per
16-repeat cell: 2 s for the counts-level stacks, 12 s for probability ZNE, 40 s for folding,
45 s with a twin.)

## 3. Reading

* **Readout correction + symmetry filtering is the stack for this model at `p_e = 1e-3`** (for
  stronger gate noise see Section 5): bias +0.3 % or less
  in every column (gates +0.40 ± 0.12 % is the largest, statistically marginal), *no* increase in
  the spread (0.096 versus 0.113 unmitigated: the filter even discards some noisy shots), no extra
  circuits per energy, only the 32,768 one-off calibration shots. Each part works on its own
  error source (readout: readout column; symmetry: most of the gates column), so nothing cancels
  by accident -- the M2 criterion.
* **Why symmetry is so effective here, unlike on FakeGuadalupeV2.** The PRB 106 channels are Pauli
  errors: in any measurement basis two of the three Paulis flip the measured spin and move the
  shot out of the `n/2` sector, so the filter catches most single errors. On the compiled device
  model (M2) much of the gate noise stayed inside the singlet sector and left a 4-5 % residual.
* **ZNE alone does not see readout error** (+7.6 % under full noise) and costs 3x the circuits
  and 2-3x the spread; with readout correction it is unbiased (-0.3 %) at 2.6x the spread of
  readout+symmetry. Layer folding has a smaller spread than probability scaling (longer lever
  arm), at 3-5x deeper circuits.
* **Symmetry + ZNE does not over-correct badly here** (-0.4 to -0.8 %; M2 found -7 % on the
  device). The filtered energy is no longer exponential in the noise scale, so the fit is
  slightly off, but the remaining noise is small.
* **Twin rescaling** removes the bias on its own (-0.5 %) because layer noise damps all states
  nearly uniformly, but its one-off twin measurement adds a run-wide random factor: the spread
  between fresh setups doubles (0.25). Within one run that factor is common to every energy.
* Relative to the M2 choice (readout + symmetry + twin): adding the twin after readout +
  symmetry changes the bias by less than its own error and doubles the setup-to-setup spread.
  On this model the twin is not needed.

## 4. Infinite-shot limits: is each bias the method's or the sampling's?

`benchmarks/m2_exact_limits.py`, `results/20261117T024920Z-graph_hva_m2_exact` (data:
[`data/graph_hva_m2_exact_limits.csv`](data/graph_hva_m2_exact_limits.csv),
[`data/graph_hva_m2_exact_damping.csv`](data/graph_hva_m2_exact_damping.csv)). As main's M2
did for FakeGuadalupeV2 (its "Why readout plus symmetry looks accurate" and "Why the filter
rescues the twin"), every stack's **infinite-shot value** is computed from exact density
matrices: the outcome distribution of each measurement basis, the readout channel applied
analytically, the estimators of `LogicalNoiseEnergy` evaluated on the distribution (same
correct-then-filter ratio and delta-method variance), ZNE by the same weighted exponential
fit at the exact `E(lambda)`, twin rescaling with the exact twin energy. The same code gives
the standard error a 1024-shot estimate should have (with the twin's 16384-shot measurement
noise), and the result for **filter-then-correct**, the order main's M2 withdrew (not
implemented as a stack; with symmetric flips it divides the filtered energy by `(1 - 2q)^2`).
Not modelled: the finite readout calibration (about 0.1 % of the energy between setups).

**Every sampled cell agrees with its limit.** At `p_e = 1e-3` (72 cells): `|z| <= 2.2`, where
`z` = (16-repeat mean - limit) / SEM; the observed spread between fresh setups is 0.7-1.5 times
the predicted standard error (a standard deviation from 16 values is itself uncertain by about
18 %). At stronger noise (Section 5) the largest `|z|` is about 3, in the twin rows, which share
their per-repeat twin draws (one fluctuation counted in four rows, as in main's M2 seed check),
and in the counts-level rows at `1e-2`, which share their samples. A fresh-seed check there
(4 x 16384 shots per basis) agrees with the limits: no mitigation +26.70 +- 0.08 % (limit
+26.65 %), symmetry +10.44 +- 0.19 % (limit +10.37 %). So the table of Section 2 measures the methods, not
sampling luck: readout + symmetry converges to +0.39 % (opt, full noise), the twin stacks to
-0.24 to -0.29 %, ZNE without readout correction to +7.8 %.

**The order of the counts layers matters, as on main.** Filter-then-correct over-corrects by
-7.7 % with readout error alone and by -6.8 % under full `1e-3` noise; correct-then-filter is
exact with readout error alone (Section 2, `-0.00 %` in the limit). At `p_e = 1e-2` the
withdrawn order looks *better* (+7.0 % against +10.4 %), because its over-correction now cancels
part of a large gate residual: main's FakeGuadalupeV2 cancellation, reproduced on this model at
a stronger noise.

**Why the twin works only with the filter (damping `D = E_noisy / E_ideal`):**

| noise | estimator | target (opt) | target (0.3) | twin |
|---|---|---|---|---|
| gates 1e-3 | none | 0.9697 | 0.9705 | 0.9669 |
| gates 1e-3 | readout + symmetry | 0.9961 | 0.9967 | 0.9937 |
| gates 1e-2 | none | 0.7335 | 0.7405 | 0.7157 |
| gates 1e-2 | readout + symmetry | 0.8963 | 0.9085 | 0.9049 |

Without the filter the twin is damped *more* than the targets, so rescaling overshoots (-0.3 %
at `1e-3`, -2.5 % at `1e-2`); after correction and filtering the damping is nearly uniform, and
at `1e-2` the twin is damped slightly *less* than the optimum, leaving a small positive residual
(+0.95 %) -- the same mechanism and sign as main's M2 ("the twin is damped slightly less than the
targets").

## 5. Stronger gate noise: which stack survives?

`configs/mitigation/graph_hva_m2_fixed_strong.toml` (`results/20261117T015141Z-graph_hva_m2_fixed_strong` and
`results/20261117T030218Z-graph_hva_m2_fixed_strong` for the 16 cells the first process did not
finish within its time limit; data: [`data/graph_hva_m2_fixed_strong_points.csv`](data/graph_hva_m2_fixed_strong_points.csv)),
at the optimum, 16 fresh-setup repeats, 1024 shots per basis. Infinite-shot bias in % (sampled
16-repeat mean in brackets) and the predicted 1024-shot standard error under the strongest
noise:

| stack | full 1e-3 | gates 3e-3 | full 3e-3 | gates 1e-2 | full 1e-2 | se (full 1e-2) |
|---|---|---|---|---|---|---|
| none | +10.63 (+10.58) | +8.84 (+9.05) | +15.98 (+16.33) | +26.65 (+26.31) | +32.40 (+32.11) | 0.17 |
| readout | +3.03 (+2.99) | +8.84 (+9.05) | +8.84 (+9.24) | +26.65 (+26.31) | +26.65 (+26.35) | 0.18 |
| symmetry | +1.57 (+1.48) | +1.66 (+1.89) | +3.73 (+3.87) | +10.37 (+9.89) | +14.33 (+13.67) | 0.22 |
| readout+symmetry | +0.39 (+0.27) | +1.66 (+1.89) | +1.66 (+1.75) | +10.37 (+9.89) | +10.37 (+9.61) | 0.25 |
| filter, then correct (main's withdrawn order) | -6.80 | +1.66 | -4.46 | +10.37 | +7.04 | 0.23 |
| zne_prob | +7.83 (+7.59) | -0.06 (-0.40) | +7.78 (+7.61) | -0.64 (-1.07) | +7.25 (+6.58) | 0.42 |
| zne_fold | +7.88 (+7.59) | +0.12 (-0.22) | +7.95 (+7.79) | +0.40 (+0.27) | +8.21 (+7.94) | 0.35 |
| readout+zne_prob | -0.01 (-0.30) | -0.06 (-0.40) | -0.06 (-0.25) | -0.64 (-1.07) | -0.65 (-1.36) | 0.46 |
| readout+symmetry+zne_prob | -0.28 (-0.77) | -2.17 (-2.43) | -2.15 (-2.54) | -10.64 (-11.03) | -10.65 (-10.73) | 0.59 |
| twin | -0.29 (-0.66) | -0.84 (-1.30) | -0.84 (-1.61) | -2.49 (-2.50) | -2.49 (-2.42) | 0.30 |
| readout+twin | -0.29 (-0.67) | -0.84 (-1.30) | -0.84 (-1.63) | -2.49 (-2.50) | -2.49 (-2.43) | 0.30 |
| symmetry+twin | +0.16 (+0.18) | -0.49 (-0.88) | +0.25 (-0.04) | +0.95 (+0.69) | +2.68 (+2.21) | 0.29 |
| readout+symmetry+twin | -0.24 (-0.15) | -0.49 (-0.88) | -0.49 (-0.71) | +0.95 (+0.69) | +0.95 (+0.33) | 0.33 |

* **Readout + symmetry stops being enough** as the gate noise grows: +0.4 % at `1e-3`, +1.7 % at
  `3e-3`, +10.4 % at `1e-2` (the filter keeps 92 %, 78 %, 51 % of the weight: more errors
  stay inside the singlet sector, and more shots are lost).
* **Symmetry and ZNE must not be combined** (main's M2 rule): -0.3 % at `1e-3` but -2.2 % at
  `3e-3` and -10.6 % at `1e-2`. The filtered energy is not exponential in the noise scale.
* **Readout + ZNE stays unbiased** at every strength (0.0 % to -0.7 %), at 3x the circuits and
  about 2x the spread of readout + symmetry.
* **Readout + symmetry + twin** (main's M2 choice) keeps the bias within 1 % up to `1e-2`
  (-0.2 %, -0.5 %, +1.0 %) at no extra circuits per energy; it is the stack to use when the
  gate noise is not small, with readout + ZNE as the fallback without a symmetry.
* At the PRB 106 default `p_e = 1e-3` readout + symmetry alone suffices (Section 3).

## 6. Full optimisations with the M2 stacks

Under depolarising `1e-3` + readout error 0.02, seeds 1-3; final noise-free relative error of the
returned parameters (mean +- sd over seeds) and the measured final energy minus the noise-free
energy of the same parameters (16384-shot re-measurement through the run's stack):

| optimiser (config) | stack | shots | noise-free error | measured - noise-free |
|---|---|---|---|---|
| parameter-shift GD, 60 iterations (`graph_hva_m2_gd.toml`) | none | 12.0M | 2.22 +- 0.10 % | +2.04 |
| | readout + symmetry | 12.1M | 2.07 +- 0.21 % | +0.05 |
| calibrated SPSA, step cap 0.2 (`graph_hva_m2_spsa.toml`) | none | 4.65M | 2.62 +- 0.36 % | +2.03 |
| | readout + symmetry | 4.65M | 2.66 +- 0.09 % | +0.06 |
| | readout + symmetry + twin | 4.65M | 2.44 +- 0.31 % | +0.01 |

(`results/20261116T194536Z-graph_hva_m2_gd`, `results/20261117T015513Z-graph_hva_m2_spsa`.)
With both optimisers the mitigation changes **what the run reports** (the measured energy is
10 % too high without it, correct with it), not **where the optimiser goes**: the noise-free
errors agree within the seed scatter. The bias is nearly parameter independent (Section 4:
`D` differs by < 0.1 % between the optimum and a generic point at `1e-3`), so it barely moves
the minimum, while the shot noise -- which no stack reduces -- limits the optimisers.

## 7. Consequences for optimisation and for PIDN

Every technique above is a **bias** correction. None reduces shot noise; ZNE and the twin
increase it, and readout correction slightly (a `1/(1 - 2q)^2` weight on two-body terms). The
optimisation studies found that shot noise, not bias, limits gradient descent at `p_e = 1e-3`
([`graph_hva_mitigation.md`](graph_hva_mitigation.md), Sec. 3) and that PIDN fails because it
memorises shot noise in its training gradients
([`graph_hva_pidn.md`](graph_hva_pidn.md), branch `feature/graph-hva-pidn`). Whether an M2 stack
changes that is tested on the branch `feature/graph-hva-pidn-m2`. Section 6 shows the same
for the optimisers: the stacks fix the reported energy, not the optimisation.

## 8. Reproduction

```bash
uv run kagome-vqe mitigation-study configs/mitigation/graph_hva_m2_fixed.toml --dry-run
uv run kagome-vqe mitigation-study configs/mitigation/graph_hva_m2_fixed.toml --workers 4   # ~6 min
uv run kagome-vqe mitigation-summary results/<ts>-graph_hva_m2_fixed
# stronger noise (Section 5; ~40 min with 4 workers on a loaded machine)
uv run kagome-vqe mitigation-study configs/mitigation/graph_hva_m2_fixed_strong.toml --workers 8
# infinite-shot limits (Section 4): fill the density-matrix cache, one process per point and p
for pt in opt generic0.3 twin; do for p in 1e-3 3e-3 1e-2; do
  uv run python benchmarks/m2_exact_limits.py configs/mitigation/graph_hva_m2_fixed.toml \
      --fill-only $p --points $pt &
done; done; wait
uv run python benchmarks/m2_exact_limits.py configs/mitigation/graph_hva_m2_fixed.toml \
    --sampled results/<ts>-graph_hva_m2_fixed results/<ts>-graph_hva_m2_fixed_strong
# full optimisations (Section 6): one process per run
uv run kagome-vqe study configs/studies/graph_hva_m2_spsa.toml --only <run> --no-summary
uv run kagome-vqe study configs/studies/graph_hva_m2_gd.toml --only <run> --no-summary
```

The `opt` point needs the noise-free optimum of `configs/studies/graph_hva_optimizer_representative.toml`
(`results/20261111T132627Z-graph_hva_opt_repr`, local).

## 9. Limitations

* One lattice, one depth, one readout error; three gate-noise strengths (the stronger ones at
  the optimum only), 16 repeats; the readout error is a uniform symmetric flip (no asymmetry or
  crosstalk). The infinite-shot limits assume an exact readout calibration.
* Full optimisations: three seeds per stack, one learning rate / SPSA setting.
* The paper-layer model is Pauli noise on a logical circuit; on compiled hardware noise the
  ranking can differ (it did in M2: symmetry left 4-5 %).
* Readout correction on the full distribution and the automatic twin need classical simulation
  (about 20 qubits), as in M2.
