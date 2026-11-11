# Generalised lattices, the graph-based HVA, and hardware mapping

This document describes the `graph_hva` ansatz, the configurable lattice layer, and the two
paper-inspired hardware mappings. Everything here is additive: the ring ansatzes `hva` and
`rvb`, the 12-site defaults and every existing config behave exactly as before.

Experiments with these tools are written up separately: the L-BFGS-B depth sweep in
[`experiments/graph_hva_baseline.md`](experiments/graph_hva_baseline.md), the optimiser
comparison in [`experiments/graph_hva_optimizers.md`](experiments/graph_hva_optimizers.md)
and the compile-only mapping benchmarks in [`hardware.md`](hardware.md).

References (PDFs in `references/KagomeHVA/`):

- **PRB 105** - J. L. Bosse and A. Montanaro, *Probing ground-state properties of the kagome
  antiferromagnetic Heisenberg model using the variational quantum eigensolver*,
  Phys. Rev. B 105, 094409 (2022).
- **PRB 106** - J. Kattemolle and J. van Wezel, *Variational quantum eigensolver for the
  Heisenberg antiferromagnet on the kagome lattice*, Phys. Rev. B 106, 214429 (2022).

## 1. Architecture

```
lattice.py / kagome_lattices.py   sites, bonds, couplings, coordinates, boundary, dimers
          |
hamiltonian.py                    H = sum J_ij (XX + YY + ZZ)          (unchanged)
          |
schedule.py                       ordered layers of disjoint bonds (logical gate order)
          |
ansatz/graph_hva.py               singlets + p cycles of exchange gates (OPG), optional H0
          |                       -- hardware-independent logical circuit --
mapping/strategies.py             prb105 | prb106_grid | generic   -> routed circuit + permutation
mapping/core.py                   block parser, SWAP router, 3-CNOT synthesis, verification
mapping/placement.py              embed the routed grid into a backend coupling map (VF2)
          |
transpile.py (compile_mapping)    native gates, routing disabled, observable follows layout
          |
vqe.py / estimators.py            unchanged optimisation, estimators, mitigation, results
```

Responsibilities do not leak: the lattice knows nothing about qubits on hardware, the ansatz
knows nothing about SWAPs, and a mapping must reproduce the ansatz's gate order exactly.

`vqe.compile_problem` is the single dispatch point. The ring and library ansatzes with
`mapping.strategy = "generic"` (the default) take exactly the code path they always took.

## 2. Lattices

A `Lattice` is a validated, immutable bond list. It may carry metadata:

| field | meaning |
|---|---|
| `site_order` | ansatz qubit `k` hosts site `site_order[k]` (as before) |
| `coords` | integer triangular-lattice coordinates `(r, c)` per site |
| `boundary` | `"open"`, `"periodic"` or `"unspecified"` |
| `wrapped_bonds` | bonds that cross a periodic boundary |
| `dimers` | preferred dimer covering (perfect matching of bonds) |

**Coordinates.** The kagome lattice is the triangular lattice (bonds `(0,1)`, `(1,0)`, `(1,1)`)
with every site at odd `r` *and* odd `c` removed; those holes are hexagon centres. The bond list
alone defines the physics; coordinates are only used by the geometric mappings and plots. The
number of sites never determines a geometry.

**Validation.** It covers site ranges, self-loops, duplicate bonds (in either orientation),
zero or non-finite couplings, coordinates on holes, wrapped bonds without a periodic
boundary, and dimer coverings. A covering must be made of bonds and cover each site exactly
once. Generators and custom graphs also reject disconnected graphs, and custom coordinates
must make every bond a nearest neighbour.

### Generators (`[lattice] kind = ...`)

| kind | sites / bonds | notes |
|---|---|---|
| `kagome_unit_cell` | 12 / 18 | the original Star of David; legacy labels, Hamiltonian-cycle order, spoke covering |
| `kagome_kvqe20` | 20 / 30 | PRB 106 Fig. 2 open patch with the paper's dimer covering |
| `kagome_bm_rectangle` | `rows*cols` | graph induced by the PRB 105 embedding on a qubit rectangle (their Fig. 4: 2x4 ... 3x8); covering = colour-1 pairs for even `cols` |
| `kagome_patch` | any | all kagome sites in a `rows x cols` window of the triangular lattice, minus `drop` |
| `kagome_periodic` | `3*rows*cols` / `6*rows*cols` | torus of unit cells, `rows, cols >= 2`; wrapped bonds recorded |
| `custom` | explicit | `num_sites`, `bonds = [[a, b], [a, b, J], ...]`, optional `coords`, `boundary`, `wrapped_bonds` |

Sanity anchors (tested): the unit cell keeps its 54-term Hamiltonian and spectrum
`[-18, -18, -16.961]`. The 12-site torus has `E0/N = -0.45374` in `S.S` units.

### Dimer coverings

`[lattice] dimers = "default" | "auto" | [[a, b], ...]`. `"default"` uses the generator's
covering if it has one, else the lexicographically first perfect matching. That matching comes
from a deterministic depth-first search: cover the lowest free site with its smallest free
neighbour. Graphs without a perfect matching, odd site counts included, are rejected with an
error. There is no silent fallback to another reference state.

```toml
[lattice]
kind = "custom"
num_sites = 6
bonds = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [2, 3, 0.5]]
dimers = [[0, 1], [2, 3], [4, 5]]
```

`kagome-vqe lattice CONFIG --plot lattice.png` prints the resolved graph and draws it, with
bonds coloured by schedule layer and dimers shaded (`kagome_vqe.visualize`).

## 3. The graph-based HVA (`[ansatz] type = "graph_hva"`)

### Conventions

| quantity | definition |
|---|---|
| Hamiltonian | `H = sum_<ij> J_ij (X_i X_j + Y_i Y_j + Z_i Z_j) = 4 sum J_ij S_i.S_j` (Pauli normalisation, as everywhere in the package) |
| exchange gate | `G_ij(theta) = exp(-i theta/2 (XX + YY + ZZ)) = exp(-2 i theta S_i.S_j) = RZZ(theta) RYY(theta) RXX(theta)` |
| relation to PRB 106 | `HEIS(alpha) ~ exp(-i alpha S.S)`, so `alpha = 2 theta` |
| relation to PRB 105 | `exp(-i t X.X)` with Pauli `X.X`, so `t = theta / 2` |
| couplings | `J_ij` is absorbed into `theta`; the ansatz uses the bond graph, not the weights |
| reference state | singlets `(|01> - |10>)/sqrt 2` on the dimer covering `D` |
| `H0` | `sum_{(ij) in D} (XX + YY + ZZ)_ij`; the reference state is its ground state |

The full Heisenberg interaction is one SU(2)-symmetric gate. There are never separate XX, YY or
ZZ parameters, so the state stays in the total-spin singlet sector (`S^2 = S_z = 0`, tested for
random parameters).

### Circuit, chronologically

```
singlets on D
for c = 1 .. p:
    [H0 block      if include_h0 and h0_position = "start"]
    for layer in schedule.layers:          # disjoint bonds, parallel
        G_bond(theta[c, bond])  for bond in layer
    [H0 block      if include_h0 and h0_position = "end"]      (default position)
```

H0 block: `exp(-i phi_c/2 H0) = prod_{(ij) in D} G_ij(phi_c)`. It has one parameter per cycle,
never shared with a bond gate.

### One parameter per gate (OPG)

Every exchange gate has its own parameter: `N_theta = p * N_b` without H0 and
`p * (N_b + 1)` with H0. `parameterization = "per_layer"` (PRB 105's "per edge colour") is the
only form of sharing, and it is explicit opt-in. Parameter names `tCCC_KKKK` (cycle, slot)
sort alphabetically in chronological order, which is the order of the parameter vector.
Parameter-shift gradients are exact: each exchange gate is three commuting rotations, and
`unshare_parameters` handles them (tested against finite differences).

### Gates that only add a phase, and redundant H0 parameters

- In cycle 1, a gate on a reference dimer whose sites are still untouched acts on an exact
  singlet: `G(theta)|s> = e^{3 i theta/2} |s>`. The H0 block at the *start* of cycle 1 is
  entirely of this kind. `phase_only_gates` reports them. `prune_phase_gates = true` removes
  them and drops any parameter left without a gate; the state does not change, and
  `num_parameters` versus `nominal_parameters` reflects the drop.
- An H0 gate on dimer `(a, b)` *fuses* with a bond gate on the same pair if no gate touching
  `a` or `b` sits between them: `G(phi) G(theta) = G(phi + theta)`. If a whole H0 block fuses,
  its parameter is a gauge freedom. `redundant_parameters` reports this; it does not prune.

These diagnostics explain the order in PRB 105's Eq. (4),
`U = prod_i [e^{-i theta_0 H0} U3 U2 U1 U5 U4]`. Read as an operator product (rightmost first)
it is chronologically `H4, H5, H1, H2, H3, H0`. With `H0 = H1`, this is the only ordering we
found in which no cycle-1 gate is phase-only and H0 never fuses with H1. In the listing order
of their Fig. 3 (`H1 ... H5`), all cycle-1 `H1` gates are phase-only, and an end-of-cycle H0 is
redundant (tested). Therefore:

- `schedule = "prb105"` uses the Eq. (4) order, and `"prb105_fig3"` the Fig. 3 order;
- `h0_position` defaults to `"end"`;
- the `greedy` schedule orders its colour classes so that dimer-rich layers come last
  (no H0) or in the middle (with H0). This gives zero phase-only gates and no redundant H0 on
  the shipped lattices (tested).

### Schedules (`[ansatz] schedule = ...`)

| schedule | layers | needs |
|---|---|---|
| `greedy` | minimal deterministic edge colouring (4 layers on all shipped lattices, the minimum; PRB 105 App. C.3) | any graph |
| `prb105` / `prb105_fig3` | `H1..H5` of PRB 105 Fig. 2 | open patch with coordinates |
| `prb106` | the six layers of PRB 106 Fig. 2 (orientation x parity rules; tested against the figure bond by bond) | open patch with coordinates |
| `explicit` | `layers = [[[a, b], ...], ...]` | validated: matchings, every bond exactly once |

`layer_order = [...]` permutes any schedule.

## 4. Hardware mapping (`[mapping]`)

```toml
[mapping]
strategy = "prb105"        # "generic" (default) | "prb105" | "prb106_grid"
fallback = "error"         # or "generic": allowed, and then recorded as generic_fallback
physical_layout = []       # optional hardware qubit per routed-circuit qubit
restore_at_end = false     # undo outstanding SWAPs at the end
merge_swaps = true         # G(theta) SWAP = G(theta + pi/2)
exchange_synthesis = "cx3" # 3-CNOT Heisenberg gate (Vatan-Williams), or "rotations"
```

### How routing works

1. The logical circuit is split into blocks; consecutive `rzz/ryy/rxx` on one pair form one
   exchange block, also when parameter shift has split the angles.
2. Each logical qubit gets a *home* position on an abstract square grid; this is the
   strategy's embedding.
3. Blocks are walked **in their original order**. If a block's qubits are not grid
   neighbours, the strategy proposes SWAPs that make them neighbours, and conflicting earlier
   SWAPs are undone. SWAPs are undone lazily, only when a later gate needs the original
   arrangement. SWAPs between two qubits that are both still untouched `|0>` are elided.
   Order is preserved, so the routed circuit implements the logical unitary exactly, up to
   the recorded final permutation.
4. Optionally, a SWAP next to an exchange block on the same pair is merged into it.
5. The routed grid is placed on the device by subgraph monomorphism into the backend coupling
   map (rustworkx VF2). Among the candidates, the one with the lowest calibrated error wins:
   two-qubit error times usage, plus readout error. A user `physical_layout` is validated
   instead. Missing couplers or qubits only reduce the candidates; the device is never assumed
   to be a perfect grid.
6. Qiskit compiles to native gates with `routing_method = "none"`, so the strategy's SWAPs are
   the only ones. The observable follows the final permutation and the transpiler layout, and
   `IsaProblem.physical_qubits` records where each logical qubit is measured. Counts-level
   mitigation uses that field.

Auxiliary qubits (stations, or empty grid positions) start in `|0>`. They are moved by SWAPs
and never acted on otherwise, so they stay `|0>` (checked by `verify_routed`).

### Strategy A: `prb105` (Bosse & Montanaro)

The embedding has no auxiliary qubits. Grid row `g` holds the horizontal kagome line
`r = 2g`, with the apex sites of row `r - 1` inserted between their two lower neighbours
(`bm_coord_to_grid`). Two thirds of the bonds are grid edges: groups `H1`, `H2` (horizontal)
and `H3` (vertical). `H4` (apex to upper-right) and `H5` (across an apex) become grid edges
after swapping every apex with its right-hand neighbour. That is the SWAP layer of their
Fig. 3, and a second SWAP layer brings the qubits back.

### Strategy B: `prb106_grid` (Kattemolle & van Wezel, KVQE_G)

The sheared triangular lattice is laid on the grid as is. The hexagon centres (odd, odd) are
swapping stations: auxiliary qubits in `|0>`. A diagonal kagome bond becomes a grid edge after
its lower-right endpoint is swapped into the adjacent station. One auxiliary qubit is needed
per three sites in the bulk.

### Paper-exact, paper-inspired, fallback

| status | meaning |
|---|---|
| `paper_exact` | the paper's lattice **and** schedule; the routed circuit is the paper's construction |
| `paper_inspired` | the same embedding on another open patch or with another schedule; the paper's depth is not implied |
| `generic` | Qiskit layout + SABRE routing (exchange gates pre-synthesised with 3 CNOTs) |
| `generic_fallback` | a geometric strategy was requested, did not apply, and `fallback = "generic"` allowed the substitution |

What the tests check against the papers (counts are two-qubit layers with exchange and SWAP
gates as single operations, the papers' convention):

| construction | paper | reproduced |
|---|---|---|
| PRB 106, 20-site patch | 24 grid qubits, 4 stations, 30 HEIS + 16 SWAP per cycle, depth 6 per cycle | yes, steady state (+6 per cycle, +16 SWAPs) |
| PRB 105, 3x8 / 3x6 / 2x10 rectangles | no auxiliaries, 7 layers per cycle | yes, steady state (+7 per cycle) |

Supported geometries: any open kagome patch with coordinates, for both strategies. Not
supported: periodic lattices (wrapped bonds are not grid-local), custom graphs without
coordinates, and devices whose coupling map does not contain the routed grid, e.g. heavy-hex
Heron. These are reported as `MappingError`, or as a recorded generic fallback.

## 5. Backends and dry runs

`backend.kind` is unchanged: `statevector`, `aer_ideal`, `aer_noisy`, `fake`, `ibm`. With a
geometric mapping and `statevector`, the routed circuit itself is simulated, including
stations. This is a full correctness check but costs `2^(sites + stations)` amplitudes.

Compile-only commands (no job is ever submitted):

```bash
kagome-vqe run CONFIG --dry-run                            # summary incl. mapping status and resources
kagome-vqe map CONFIG                                      # prb105 vs prb106_grid vs generic table
kagome-vqe map CONFIG -b aer_noisy:FakeNighthawk           # Nighthawk (square grid) fake backend
kagome-vqe map CONFIG -b ibm:ibm_phoenix                   # real device Target (credentials), still compile-only
kagome-vqe map CONFIG -b aer_noisy:FakeBoston              # Heron heavy-hex: grid strategies reported unsupported
```

## 6. Known limitations

- **Initial singlets on diagonal dimers** (PRB 106 patch): our router prepares them through
  the station and restores the station afterwards. This costs two extra two-qubit layers once,
  at the start, compared with the paper's quoted total `3 + 6p`. The steady-state cycle
  matches the paper.
- With an H0 block, a PRB 105 cycle is 8 layers rather than 7, because H0 is a genuine extra
  layer. PRB 105's depth-7 figure refers to the cycle without H0.
- `prb105` on non-rectangular patches may use empty grid positions as auxiliary qubits.
- The geometric strategies need a square-grid subgraph on the device. On heavy-hex they are
  unsupported by design; use `generic`.
- `generic` SWAP counts in `kagome-vqe map` come from a level-1 SABRE run with the same seed.
  At level 3, Qiskit may absorb some of them.
- Theoretical depth is not hardware performance: the comparison tables report qubits, CZ
  count and depth for one device and seed only.
- Exact diagonalisation in `prepare` is sparse Lanczos: about a minute at 20 sites, infeasible
  much beyond 24.
