# graph_hva error mitigation, M2 style: readout, symmetry, ZNE and twin on layer noise

Status (2026-11-17): the full M2 comparison of the legacy study
([`legacy_m0_m1_m2.md`](legacy_m0_m1_m2.md), "Error mitigation (roadmap M2)") redone for the
graph HVA on PRB 106 paper-layer noise, at fixed parameters. ZNE alone, its methods (probability
scaling, layer folding, fits) and ZNE inside gradient descent are in
[`graph_hva_mitigation.md`](graph_hva_mitigation.md); this document compares all four M2
techniques and their combinations.

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

* **Readout correction + symmetry filtering is the stack for this model**: bias +0.3 % or less
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

## 4. Consequences for optimisation and for PIDN

Every technique above is a **bias** correction. None reduces shot noise; ZNE and the twin
increase it, and readout correction slightly (a `1/(1 - 2q)^2` weight on two-body terms). The
optimisation studies found that shot noise, not bias, limits gradient descent at `p_e = 1e-3`
([`graph_hva_mitigation.md`](graph_hva_mitigation.md), Sec. 3) and that PIDN fails because it
memorises shot noise in its training gradients
([`graph_hva_pidn.md`](graph_hva_pidn.md), branch `feature/graph-hva-pidn`). Whether an M2 stack
changes that is tested on the branch `feature/graph-hva-pidn-m2`.

## 5. Reproduction

```bash
uv run kagome-vqe mitigation-study configs/mitigation/graph_hva_m2_fixed.toml --dry-run
uv run kagome-vqe mitigation-study configs/mitigation/graph_hva_m2_fixed.toml --workers 4   # ~6 min
uv run kagome-vqe mitigation-summary results/<ts>-graph_hva_m2_fixed
```

The `opt` point needs the noise-free optimum of `configs/studies/graph_hva_optimizer_representative.toml`
(`results/20261111T132627Z-graph_hva_opt_repr`, local).

## 6. Limitations

* One lattice, one depth, one gate-noise strength and one readout error, two parameter points,
  16 repeats; the readout error is a uniform symmetric flip (no asymmetry or crosstalk).
* The paper-layer model is Pauli noise on a logical circuit; on compiled hardware noise the
  ranking can differ (it did in M2: symmetry left 4-5 %).
* Readout correction on the full distribution and the automatic twin need classical simulation
  (about 20 qubits), as in M2.
