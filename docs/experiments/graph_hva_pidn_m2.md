# Can M2-style error mitigation stop PIDN from learning noise?

Status (2026-11-17): three-seed study on the PRB 105 2x6 rectangle (`graph_hva`, `p = 2`, 32
parameters) under PRB 106 depolarising layer noise `p_e = 1e-3` plus a readout error `q = 0.02`.
Branch `feature/graph-hva-pidn-m2` (PIDN, [`graph_hva_pidn.md`](graph_hva_pidn.md), merged with
the M2 comparison, [`graph_hva_m2_mitigation.md`](graph_hva_m2_mitigation.md)).

## 1. Question and answer

PIDN failed because its network fits the noise in the Stage I gradients
([`graph_hva_pidn.md`](graph_hva_pidn.md), Sec. 5). The M2 techniques -- readout correction,
symmetry filtering, ZNE, twin rescaling -- make the measured *costs* accurate (Sec. 2 of the M2
document: readout + symmetry, bias <= 0.4 % at no extra variance). Does feeding PIDN mitigated
data fix it?

**No.** Mitigation removes bias; PIDN is limited by variance, and by what variance does to the
*parameter path* of Stage I. Concretely, on this problem:

1. Every M2 stack leaves the gradient noise unchanged (readout + symmetry, twin) or larger (ZNE);
   only more shots reduce it.
2. Even with *exact* targets, a surrogate trained on a noisy Stage I path cannot predict the
   next gradients: noisy steps knock the parameters off the descent path, and the true gradient
   at the next point is then dominated by curvature times a random displacement.
3. With a Stage I run through readout + symmetry at 8x the shots (a much smoother path), PIDN
   still fails in closed loop. With an M2-measured gradient every 5 steps it is as good as, but
   no better than, plain gradient descent with the same number of measured gradients.

## 2. Experiments

All under the "full" noise model of the M2 table (depolarising `1e-3` + readout `0.02`), 1024
shots per basis unless noted, seeds 1-3, `t_init = 10`, 60 iterations in total.

| step | config | what varies | results |
|---|---|---|---|
| A. relabelled Stage I | `configs/pidn/graph_hva_pidn_m2_relabel.toml`, `kagome-vqe pidn-relabel` | the Stage I *targets* (cost and parameter-shift gradient at the 10 recorded points of the ZNE runs) through 6 pipelines; the path is fixed | `results/20261116T183018Z-graph_hva_pidn_m2_relabel` |
| B. control | `benchmarks/pidn_exact_targets_control.py` | exact noise-free targets on the same recorded noisy paths | [`data/graph_hva_pidn_exact_targets_control.csv`](data/graph_hva_pidn_exact_targets_control.csv) |
| C. real M2 Stage I | `configs/studies/graph_hva_m2_gd.toml` | gradient descent run through the pipeline (so the path reflects it): unmitigated and readout + symmetry, 60 iterations; readout + symmetry at 8x shots, 12 iterations | `results/20261116T194536Z-graph_hva_m2_gd` |
| D. PIDN closed loop | `configs/pidn/graph_hva_pidn_m2.toml` (on A), `graph_hva_pidn_m2_path_rs.toml`, `graph_hva_pidn_m2_path_rs_x8.toml` (on C) | `pidn_paper`, `pidn_small`, `pidn_anchor5` (a measured gradient through the same pipeline every 5 steps, then retraining), `last_gradient` | `results/20261116T185211Z-graph_hva_pidn_m2`, `...T200642Z-..._path_rs`, `...T201708Z-..._path_rs_x8` |

Pipelines of A: `raw` (none), `rs` (readout + symmetry), `rst` (+ twin), `rzne` (readout + ZNE,
probability scaling 1, 2, 3, exponential), `rs_x4`, `rs_x8` (readout + symmetry with 4096 / 8192
shots per basis). A 16x pipeline was started first and did not finish inside the 40-minute
per-process cap; it was replaced by 8x (the `relabel.toml` stored in the results directory still
lists it).

## 3. Results

### 3.1 What mitigation does to the Stage I targets (A, C)

Over the 10 Stage I points, means over seeds. `cos` / `cos late`: cosine of the measured gradient
with the exact noise-free gradient (all points / last 5); `|g - g_ideal|`: gradient error norm;
`SNR late`: noise-free gradient norm / gradient error over the last 5 points; cost error:
measured cost minus noise-free energy.

| pipeline | Stage I circuits (shots per basis) | cos | cos late | abs(g - g_ideal) | SNR late | cost error |
|---|---|---|---|---|---|---|
| none | 1,950 (1024) | 0.79 | 0.68 | 1.13 | 1.1 | **+1.99** |
| readout + symmetry | 1,950 (1024) | 0.80 | 0.70 | 1.19 | 1.1 | +0.02 |
| readout + symmetry + twin | 1,950 (1024) | 0.82 | 0.74 | 1.16 | 1.1 | -0.06 |
| readout + ZNE | 5,850 (1024) | 0.65 | 0.54 | 1.98 | 0.65 | -0.02 |
| readout + symmetry, 4x shots | 1,950 (4096) | 0.93 | 0.88 | 0.61 | 2.0 | +0.00 |
| readout + symmetry, 8x shots | 1,950 (8192) | 0.96 | 0.94 | 0.43 | 2.9 | +0.04 |
| *(C) readout + symmetry, own path* | 1,950 (1024) | 0.76 | 0.63 | 1.21 | 0.85 | +0.01 |
| *(C) readout + symmetry 8x, own path* | 1,950 (8192) | 0.90 | 0.82 | 0.43 | 1.8 | +0.05 |

Mitigation fixes the cost (+1.99 -> +0.02) and leaves the gradient error where it was; ZNE
makes it worse; shots reduce it as `1/sqrt(shots)`.

### 3.2 The surrogate cannot predict along a noisy path, even from exact targets (A, B)

Open loop: trained on the 10 points, scored on the next 1-5 recorded points; cosine of the
predicted gradient with the exact noise-free gradient, mean over seeds.

| Stage I data | `pidn_paper` | `pidn_small` | `quadratic` | `last_gradient` |
|---|---|---|---|---|
| noise-free run (smooth path), exact targets | **0.97** | 0.93 | 0.66 | 0.93 |
| ZNE run's noisy path, **exact** targets (B) | 0.27 | 0.32 | 0.37 | 0.13 |
| same path, targets from: none / readout + symmetry | 0.22 / 0.20 | 0.25 / 0.21 | 0.28 / 0.25 | 0.08 / 0.05 |
| same path, readout + symmetry + twin / readout + ZNE | 0.26 / 0.12 | 0.19 / 0.11 | 0.25 / 0.17 | 0.08 / -0.01 |
| same path, readout + symmetry 4x / 8x | 0.24 / 0.25 | 0.24 / 0.24 | 0.31 / 0.32 | 0.05 / 0.10 |
| (C) readout + symmetry 8x, own path (1-2 steps) | 0.25 | 0.15 | 0.56 | 0.22 |

Along the noisy paths the noise-free gradient has norm 0.7-1.0 and consecutive gradients have
cosine 0.1-0.4; along the noise-free path the gradient has norm ~0.2 and consecutive cosine
0.996 (control B output). Every noisy step moves the parameters off the valley floor in a random
direction; the gradient there is mostly curvature times that displacement, which no amount of
history predicts. Better targets on the same path do not help (0.20-0.26 for every pipeline,
0.27 even when exact); a smoother path (8x shots) helps a little (quadratic 0.56).

### 3.3 Closed loop (D)

Noise-free relative error of the parameters; median over seeds (worst). Error when the
surrogate takes over: 0.0274-0.0276 (A, C 1x) and 0.0265 (C 8x). `cos`: mean cosine of the
surrogate gradient with the noise-free gradient during the surrogate phase.

| Stage I | variant | circuits (shots) | final error | cos |
|---|---|---|---|---|
| A: none / readout+symmetry / +twin / readout+ZNE | `pidn_paper` | 2,100 / 2,100 / 2,100 / 6,000 | 0.55 / 0.41 / 0.44 / 0.42 | -0.37 / -0.34 / -0.47 / -0.31 |
| A: readout+symmetry 4x / 8x | `pidn_paper` | 2,100 (8.6M / 17M) | 0.31 / 0.44 | -0.44 / -0.56 |
| A: all pipelines | `pidn_small` | 2,100-6,000 | 0.22-0.54 | -0.29 to -0.35 |
| A: all pipelines | `last_gradient` | 1,950-5,850 | 0.50-0.81 | -0.29 to -0.54 |
| C: readout+symmetry, own path | `pidn_paper` / `pidn_small` / `last_gradient` | 2,100 | 0.26 / 0.25 / 0.79 | -0.46 / -0.35 / -0.40 |
| C: readout+symmetry 8x, own path | `pidn_paper` / `pidn_small` / `last_gradient` | 2,100 (17M) | 0.36 / 0.22 / 0.31 | -0.54 / -0.43 / -0.62 |
| A: none / readout+symmetry / +twin / 4x / readout+ZNE | `pidn_anchor5` | 4,050 (11,850 ZNE) | 0.024 / 0.026 / 0.026 / 0.023 / 0.027 | 0.03-0.21 |
| C: readout+symmetry / 8x, own path | `pidn_anchor5` | 4,050 (4.2M / 33M) | **0.0264 / 0.0231** | 0.09 / 0.11 |
| *gradient descent, readout+symmetry, after 20 / 60 measured gradients* | - | 3,900 / 11,700 | 0.0263 / 0.0216 | - |
| *gradient descent, unmitigated, after 20 / 60* | - | 3,900 / 11,700 | 0.0259 / 0.0219 | - |

![closed loop on the 8x readout+symmetry path](figures/pidn_m2_path_rs_x8_error.png)

* Without anchors every run is worse than at hand-over within one to three surrogate steps
  (iteration 11 in 48 of the 54 relabelled-data runs and in 17 of the 18 own-path runs, 12-13 in
  the rest). The surrogate gradient is *anti*-correlated with the true one in closed loop (-0.3 to
  -0.6) although it was weakly positive open loop (0.1-0.3): stepping along `-grad C_hat` moves
  the parameters exactly where the surrogate's errors point downhill, the classic failure of
  optimising against a learned model.
* With a measured gradient every 5 steps (`pidn_anchor5`) the error stays at 0.023-0.027 --
  the error gradient descent reaches with the same 20 measured gradients (0.026). At the anchors
  the retrained surrogate agrees with the measured gradient only at cosine ~0. The surrogate
  steps in between add nothing; the anchored 8x run costs 33M shots for 0.0231, unanchored
  gradient descent reaches 0.0216 with 12M.
* With noise-free Stage I data (the clean limit of [`graph_hva_pidn.md`](graph_hva_pidn.md),
  Sec. 5.3) the same network continues the descent for 10-15 steps. Between that and the 8x
  pipeline (gradient SNR ~3, consecutive-gradient cosine ~0.8) lies more than an order of
  magnitude in shots.

### 3.4 Gradient descent with M2 mitigation (C)

| 60 iterations, 11,700 circuits | final noise-free error (median, max) | final fidelity | final measured energy (noise-free) |
|---|---|---|---|
| unmitigated | 0.0219 (0.0233) | 0.671 | -17.42 (-19.47): bias +2.05 |
| readout + symmetry | 0.0216 (0.0221) | 0.678 | -19.45 (-19.48): bias +0.03 |

As with ZNE ([`graph_hva_mitigation.md`](graph_hva_mitigation.md), Sec. 3): mitigation does
not change how well gradient descent optimises (the readout damping rescales every gradient by
`(1 - 2q)^2`, the direction is unchanged), but it makes the reported energy right -- here at no
extra circuit cost (only 32,768 calibration shots).

## 4. Conclusions

* **M2 mitigation does not prevent PIDN from learning noise.** It removes the bias of the costs
  (the value head now learns the right energies) but not the shot noise of the gradients, and
  PIDN's gradient field is what drives Stage III. ZNE makes the gradient noise worse.
* **The binding constraint is the signal-to-noise of the whole Stage I trajectory**, not only
  of its targets: noisy steps make the true gradient along the path unpredictable from its
  history. The network becomes useful only with nearly noise-free Stage I (shots well beyond
  8x on this problem), at which point the cheaper option is to spend those shots on plain
  gradient descent.
* **What M2 mitigation is for, here:** the energy you report. Readout + symmetry is the stack of
  choice on this noise model (bias <= 0.4 %, no extra variance, no extra circuits); optimise
  with unmitigated or readout + symmetry gradients, report the readout + symmetry (or ZNE)
  energy.

Ideas that attack variance or the path rather than bias (not done): fewer parameters (per-layer
sharing, 10 instead of 32, so each gradient is cheaper to measure precisely); a Stage I with a
smaller learning rate or gradient averaging (momentum) so the path stays smooth; anchors with a
fallback rule (use the measured gradient when the surrogate disagrees with it, which here means
always); a surrogate with an explicit local Hessian model fitted to more than one trajectory.

## 5. Reproduction

```bash
uv run kagome-vqe pidn-relabel configs/pidn/graph_hva_pidn_m2_relabel.toml          # A (one process per pipeline/seed)
uv run python benchmarks/pidn_target_quality.py results/<ts>-graph_hva_pidn_m2_relabel \
    configs/pidn/graph_hva_pidn_representative.toml --only pidn_paper pidn_small quadratic last_gradient \
    --reference "zne_original=results/<ts>-graph_hva_zne_gd/runs/*dep0.001_zne-gd_ps-s*" \
                "noise_free=results/<ts>-graph_hva_zne_gd/runs/*-sv-gd_ps-s*"
uv run python benchmarks/pidn_exact_targets_control.py configs/pidn/graph_hva_pidn_representative.toml \
    results/<ts>-graph_hva_zne_gd/runs/*dep0.001_zne-gd_ps-s* results/<ts>-graph_hva_m2_gd/runs/*full_rs-gd_ps60-s* \
    --only pidn_paper pidn_small quadratic last_gradient                                # B
uv run kagome-vqe study configs/studies/graph_hva_m2_gd.toml --only <run> --no-summary  # C
uv run kagome-vqe pidn configs/pidn/graph_hva_pidn_m2.toml                               # D on A
uv run kagome-vqe pidn configs/pidn/graph_hva_pidn_m2_path_rs.toml                       # D on C
uv run kagome-vqe pidn configs/pidn/graph_hva_pidn_m2_path_rs_x8.toml
```

Cost on the shared machine (single-threaded processes, 40-minute cap each): A ~1.5-17 min per
(pipeline, seed); C ~13 min per 60-iteration run, ~25 min per 8x run; D ~2-3 min per run, ~15
min for an anchored 8x run. Not completed and omitted: the anchored runs on the relabelled 8x
data (`pidn_anchor5-rs_x8`, expensive and superseded by C).

## 6. Limitations

One lattice, depth, noise level and readout error; three seeds; one learning rate; the relabelled
experiments hold the ZNE run's path fixed (by design, and confounded by it, which is why C was
added); the readout error is symmetric and uncorrelated.
