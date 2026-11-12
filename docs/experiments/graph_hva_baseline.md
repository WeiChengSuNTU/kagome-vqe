# graph_hva baseline: L-BFGS-B depth sweep on the exact simulator

## Goal

First check that the generalised graph-based HVA (`graph_hva`, one parameter per gate)
reaches the ground state on open lattices, and how fidelity grows with the number of cycles
`p`, with and without the H0 block. A strong deterministic optimiser (L-BFGS-B with
finite-difference gradients on exact energies) is used so that the result mainly reflects the
ansatz, not the optimiser. This is not an optimiser comparison; see
[`graph_hva_optimizers.md`](graph_hva_optimizers.md).

## Conventions

* Hamiltonian `H = sum over bonds of J (XX + YY + ZZ)`, `J = 1` (Pauli normalisation,
  `= 4 sum S.S`); exact energies and ground spaces from exact diagonalisation.
* Lattices: the open 12-site star (18 bonds), the PRB 105 2x6 rectangle (open, 12 sites,
  16 bonds) and the periodic 12-site torus (`kagome_periodic`, 2x2 cells, 24 bonds). Different
  lattices have different Hamiltonians; compare rows only within a lattice.
* Ansatz: singlets on the default dimer covering, then `p` cycles of exchange gates;
  `N_theta = p * N_b`, plus `p` with H0 (at the end of each cycle). Schedules: `greedy` for
  the star and the torus, `prb105` for the 2x6 rectangle.
* Fidelity: overlap of the final state with the exact, possibly degenerate, ground space.

## Configuration

`statevector`, L-BFGS-B (`maxiter = 400`, scipy defaults otherwise), initial parameters
uniform in `[0, 0.1)`, seeds 1 and 2, no measurement budget (exact energies, no shots).

## Results

**VQE on the exact simulator** (`statevector`, L-BFGS-B, 400 iterations, initial parameters
uniform in `[0, 0.1)`, seeds 1 and 2; driver: `benchmarks/graph_hva_vqe_sweep.py`). The relative error is
for the lowest final energy. The fidelity is the overlap with the exact, possibly degenerate,
ground space from exact diagonalisation. Schedules: `greedy` for the star and the torus,
`prb105` for the 2x6 rectangle; H0 at the end of each cycle.

| lattice | H0 | p | parameters | best rel. error | best fidelity | seeds |
|---|---|---|---|---|---|---|
| 12-site star (open) | no | 1 | 18 | 1.57e-09 | 1.0000 | 2 |
| 12-site star (open) | yes | 1 | 19 | 6.65e-08 | 1.0000 | 2 |
| PRB 105 2x6 rectangle (open) | no | 1 | 16 | 7.19e-02 | 0.5105 | 2 |
| PRB 105 2x6 rectangle (open) | no | 2 | 32 | 1.26e-02 | 0.7556 | 2 |
| PRB 105 2x6 rectangle (open) | no | 3 | 48 | 1.65e-03 | 0.9957 | 2 |
| PRB 105 2x6 rectangle (open) | no | 4 | 64 | 4.87e-04 | 0.9987 | 2 |
| PRB 105 2x6 rectangle (open) | yes | 1 | 17 | 2.29e-02 | 0.7068 | 2 |
| PRB 105 2x6 rectangle (open) | yes | 2 | 34 | 6.48e-03 | 0.9789 | 2 |
| PRB 105 2x6 rectangle (open) | yes | 3 | 51 | 6.89e-04 | 0.9981 | 2 |
| PRB 105 2x6 rectangle (open) | yes | 4 | 68 | 5.07e-04 | 0.9985 | 2 |
| 12-site torus (periodic) | no | 1 | 24 | 4.53e-02 | 0.2071 | 2 |
| 12-site torus (periodic) | no | 2 | 48 | 3.15e-02 | 0.4221 | 2 |
| 12-site torus (periodic) | no | 3 | 72 | 1.94e-02 | 0.5174 | 2 |
| 12-site torus (periodic) | no | 4 | 96 | 1.63e-02 | 0.5457 | 2 |
| 12-site torus (periodic) | yes | 1 | 25 | 4.90e-02 | 0.2931 | 2 |
| 12-site torus (periodic) | yes | 2 | 50 | 2.68e-02 | 0.4451 | 2 |
| 12-site torus (periodic) | yes | 3 | 75 | 2.06e-02 | 0.5199 | 2 |
| 12-site torus (periodic) | yes | 4 | 100 | 1.36e-02 | 0.5875 | 2 |

The open lattices converge systematically with depth. On the PRB 105 2x6 rectangle the extra
H0 parameter helps at small `p` (F = 0.98 versus 0.76 at `p = 2`). For the star the reference
state is already exact, so it is a correctness check, not a test. The periodic 12-site torus is
much harder: at `p = 4` the optimiser stops at 1.4-1.6 % error with F = 0.55-0.59.
That is consistent with the energy-fidelity bound (gap 0.466), and it is a statement about
expressiveness and optimisation at this depth, not about circuit correctness. Correctness is
established separately by the routing-equivalence tests.

### Per run

All 36 runs, from the preserved raw output
[`data/graph_hva_baseline_2026-11-10.jsonl`](data/graph_hva_baseline_2026-11-10.jsonl):

| lattice | H0 | p | seed | parameters | evaluations | iterations | rel. error | fidelity |
|---|---|---|---|---|---|---|---|---|
| star | no | 1 | 1 | 18 | 432 | 14 | 1.72e-07 | 1.0000 |
| star | no | 1 | 2 | 18 | 532 | 19 | 1.57e-09 | 1.0000 |
| star | yes | 1 | 1 | 19 | 646 | 26 | 7.86e-08 | 1.0000 |
| star | yes | 1 | 2 | 19 | 621 | 21 | 6.65e-08 | 1.0000 |
| bm2x6 | no | 1 | 1 | 16 | 356 | 16 | 7.26e-02 | 0.5105 |
| bm2x6 | no | 1 | 2 | 16 | 1742 | 76 | 7.19e-02 | 0.5091 |
| bm2x6 | no | 2 | 1 | 32 | 8072 | 218 | 1.26e-02 | 0.7491 |
| bm2x6 | no | 2 | 2 | 32 | 7980 | 225 | 1.27e-02 | 0.7556 |
| bm2x6 | no | 3 | 1 | 48 | 15335 | 292 | 1.84e-03 | 0.9943 |
| bm2x6 | no | 3 | 2 | 48 | 15341 | 298 | 1.65e-03 | 0.9957 |
| bm2x6 | no | 4 | 1 | 64 | 15235 | 220 | 6.63e-04 | 0.9981 |
| bm2x6 | no | 4 | 2 | 64 | 15238 | 223 | 4.87e-04 | 0.9987 |
| bm2x6 | yes | 1 | 1 | 17 | 1134 | 54 | 2.91e-02 | 0.6237 |
| bm2x6 | yes | 1 | 2 | 17 | 2316 | 102 | 2.29e-02 | 0.7068 |
| bm2x6 | yes | 2 | 1 | 34 | 15397 | 382 | 7.80e-03 | 0.9426 |
| bm2x6 | yes | 2 | 2 | 34 | 15411 | 396 | 6.48e-03 | 0.9789 |
| bm2x6 | yes | 3 | 1 | 51 | 15303 | 275 | 7.74e-04 | 0.9979 |
| bm2x6 | yes | 3 | 2 | 51 | 15297 | 269 | 6.89e-04 | 0.9981 |
| bm2x6 | yes | 4 | 1 | 68 | 15247 | 205 | 5.07e-04 | 0.9985 |
| bm2x6 | yes | 4 | 2 | 68 | 15241 | 199 | 5.19e-04 | 0.9984 |
| torus2x2 | no | 1 | 1 | 24 | 1860 | 60 | 4.53e-02 | 0.2058 |
| torus2x2 | no | 1 | 2 | 24 | 1684 | 59 | 4.53e-02 | 0.2071 |
| torus2x2 | no | 2 | 1 | 48 | 7637 | 140 | 3.15e-02 | 0.4221 |
| torus2x2 | no | 2 | 2 | 48 | 10839 | 206 | 3.20e-02 | 0.4196 |
| torus2x2 | no | 3 | 1 | 72 | 15235 | 197 | 1.94e-02 | 0.5114 |
| torus2x2 | no | 3 | 2 | 72 | 15234 | 196 | 1.95e-02 | 0.5174 |
| torus2x2 | no | 4 | 1 | 96 | 15278 | 146 | 1.83e-02 | 0.4994 |
| torus2x2 | no | 4 | 2 | 96 | 15182 | 147 | 1.63e-02 | 0.5457 |
| torus2x2 | yes | 1 | 1 | 25 | 2635 | 87 | 4.98e-02 | 0.2585 |
| torus2x2 | yes | 1 | 2 | 25 | 3040 | 102 | 4.90e-02 | 0.2931 |
| torus2x2 | yes | 2 | 1 | 50 | 8669 | 152 | 2.68e-02 | 0.4451 |
| torus2x2 | yes | 2 | 2 | 50 | 8722 | 154 | 2.68e-02 | 0.4451 |
| torus2x2 | yes | 3 | 1 | 75 | 15227 | 179 | 2.07e-02 | 0.5199 |
| torus2x2 | yes | 3 | 2 | 75 | 15228 | 180 | 2.06e-02 | 0.5040 |
| torus2x2 | yes | 4 | 1 | 100 | 15185 | 136 | 1.36e-02 | 0.5860 |
| torus2x2 | yes | 4 | 2 | 100 | 15188 | 139 | 1.36e-02 | 0.5875 |

**Finding (2026-11-11): all `p >= 3` runs, and the 2x6 `p = 2` runs with H0, were stopped by
scipy's evaluation limit, not by convergence.** scipy's L-BFGS-B has a default `maxfun = 15000`, checked after each iteration;
every run with more than about 15,000 evaluations above ended there, well before
`maxiter = 400`. The per-run termination message was not recorded at the time, but the
evaluation counts (15,182 to 15,411) leave no other explanation. The `p >= 3` errors are
therefore upper bounds on what L-BFGS-B reaches with more evaluations (as are the 2x6
`p = 2` H0 errors), and the torus rows in
particular do not show where its optimisation saturates. The reproduction profile below sets
`maxfun = 15000` explicitly to reproduce the table; raise it for converged numbers.

The wall times in the raw file (200 to 1,700 s per run) were measured with six sweeps sharing
the machine and include the exact diagonalisation, transpilation and the fidelity; they are
not comparable to the optimiser timings recorded by the newer persistence system.

## Reproduction

The original driver printed one JSON line per run and saved nothing else: no trace, no
parameters, no `config.toml`. **Per-run trajectories and optimised parameters of the original
36 runs do not exist**; only the JSONL summary above was kept. The same driver now saves every
run in full:

```bash
# one table row group, as originally (JSONL on stdout + full results directories)
uv run python benchmarks/graph_hva_vqe_sweep.py bm2x6 noh0 prb105 1,2,3,4 > bm2x6_noh0.jsonl

# the whole 36-run table as one study (about 1-2 CPU-hours)
uv run kagome-vqe study configs/studies/graph_hva_baseline.toml
```

Both write `results/<timestamp>-<study>/runs/<timestamp>-<run>/` directories (see
[`../architecture.md`](../architecture.md)); reload the optimised parameters with
`kagome-vqe params <run dir>`. Runs are deterministic on the statevector, so a rerun with the
same package versions reproduces the energies; one bookkeeping change since the original
sweep (the scipy wrapper no longer spends an extra objective call per iteration for its
history) lowers `nfev` slightly but does not change the optimisation path.

## Interpretation and limitations

* Open lattices converge systematically with depth; the star is a correctness check (its
  dimer-product ground state is the reference state), not a test.
* Two seeds per point say little about seed-to-seed variability.
* The periodic torus result is limited by the evaluation cap as well as by depth; see the
  finding above.
