# Physics notes

Everything a developer needs to know about the model, the ansatzes, and the one subtle
decision that decides whether VQE reaches the ground state: how ansatz qubits are mapped onto
lattice sites.

## 1. The lattice

The 12-site kagome unit cell of the IBM Open Science Prize 2022 is a "star": a hexagon
(sites 0-5) with each hexagon edge capped by an outer tip (sites 6-11). The 18 bonds are

```
hexagon:  (0,1) (1,2) (2,3) (3,4) (4,5) (5,0)
spokes:   (0,6) (1,6)   (1,7) (2,7)   (2,8) (3,8)   (3,9) (4,9)   (4,10) (5,10)   (5,11) (0,11)
```

Each tip closes a triangle with one hexagon edge, giving six corner-sharing triangles. Hexagon
sites have degree 4, tips have degree 2. The lattice lives in `kagome_vqe.lattice` as an
immutable `Lattice` dataclass; `kagome_unit_cell()` builds it.

## 2. The Hamiltonian

```
H = t * sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j),   t = 1
```

`heisenberg_hamiltonian` emits three Pauli strings per bond: 54 terms, all with coefficient 1,
as a `SparsePauliOp` (`hamiltonian.py`). The original notebook built the same operator from
qiskit-nature's `SpinOp` (spin-1/2 operators `S = sigma/2`) and multiplied by 4 to undo the
`(1/2)^2` on each term; that API no longer exists, so we write the Pauli operator directly.

Exact diagonalisation (`exact_spectrum`, sparse Lanczos on the 4096 x 4096 matrix, under a
second) gives

```
[-18.0, -18.0, -16.96132284]
```

The ground state is two-fold degenerate at -18.

## 3. Why -18: an exact dimer-product ground state

A single antiferromagnetic bond `XX + YY + ZZ` has ground state the singlet
`(|01> - |10>)/sqrt(2)` with energy -3. A product of singlets on a *perfect matching* (a set of
bonds covering every site exactly once) contributes -3 per matched bond and 0 on average for
every other bond. With 12 sites that is 6 singlets, energy -18.

Because -18 is also the exact ground energy, the singlet product is an exact ground state. The
open star has exactly two perfect matchings (`perfect_matchings()`): the six spokes
`(0,6) (1,7) (2,8) (3,9) (4,10) (5,11)` and the other six spokes
`(1,6) (2,7) (3,8) (4,9) (5,10) (0,11)`. Two coverings, two-fold degeneracy.

This is a property of the *open* 12-site cell. A periodic kagome lattice has no such product
ground state, which is what makes the real material interesting.

## 4. The ring ansatzes

Both ansatzes from the paper share the same skeleton (`ansatz/base.py`, `RingAnsatz`):

1. **Initial state.** A singlet on every *even ring bond* `(0,1), (2,3), ..., (10,11)`,
   prepared by `X X, H, CX` (`singlet_pair`).
2. **Layers** on the ring `0-1-2-...-11-0`, split into odd bonds
   `(11,0) (9,10) (7,8) (5,6) (3,4) (1,2)` and even bonds (`ring_bonds`).

`RVBAnsatz` applies an eSWAP with its own parameter on every odd bond, then on every even bond:
12 parameters per layer. `HVAnsatz` applies `exp(-i phi H_odd)` then `exp(-i gamma H_even)`
(each as `RZZ RYY RXX` per bond): 2 parameters per layer.

Cost accounting for the optimisers (`optimizers/`), with `n` parameters:

| optimiser | evaluations per iteration | total for `maxiter = 100` |
|---|---|---|
| ADAM (forward differences) | `n + 1` | RVB 1301, HVA 301 (incl. final evaluation) |
| SPSA | 2 | 201 |

These match the evaluation counts in the original notebook's CSV traces (1300, 300, 200 plus
one final call).

## 5. Site order: the decision that makes or breaks the benchmark

The ring ansatzes only know about the ring. The initial singlets sit on the even ring bonds;
if those are lattice bonds, the initial state is a dimer covering and already has energy -18.
If they are not, the initial state has zero energy on the non-bond pairs and the ansatz may
never reach the ground state.

`Lattice.site_order[k]` is the lattice site hosted by ansatz qubit `k`. The default,
`KAGOME_HAMILTONIAN_CYCLE = (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11)`, is a Hamiltonian cycle
of the star: every consecutive pair is a lattice bond (`ring_bond_fraction() == 1.0`). With it:

* all 6 even ring bonds are lattice bonds (they are the six spokes: a perfect matching);
* all 6 odd ring bonds are lattice bonds too;
* the 6 remaining lattice bonds (the hexagon) are not on the ring;
* E(all parameters = 0) = -18 for both RVB and HVA.

With the identity order (`site_order = "identity"`) only 6 of 12 ring bonds and 3 of 6 singlet
pairs are lattice bonds, and E(0) = -9. This is a deliberately hard variant, kept as an option.

### What the 2022 notebook actually did

The notebook did not define a 12-site Hamiltonian and transpile it. It hand-wrote an
"inflated" 16-qubit Hamiltonian directly on `ibmq_guadalupe` physical qubits (qubits 0, 6, 9,
15 idle), kept in this package as `inflated_guadalupe_16()`:

```
(1,2) (2,3) (3,5) (5,8) (8,11) (11,14) (14,13) (13,12) (12,10) (10,7) (7,4) (4,1)   <- a 12-cycle
(4,2) (2,5) (5,11) (11,13) (13,10) (10,4)                                          <- the hexagon
```

The 12-cycle is exactly Guadalupe's heavy-hex loop. The notebook then transpiled the ring
ansatz with `initial_layout = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]`. That layout is
*not* a faithful map of the 12-site lattice onto the 16-qubit one: only 2 of the 18 bonds land
on bonds (a faithful map exists, e.g. site `i -> [2, 4, 10, 13, 11, 5, 1, 7, 12, 14, 8, 3][i]`).
Two things rescued it:

1. level-3 routing moved the logical ring onto the heavy-hex loop anyway, and
2. the V1 `Estimator` evaluated the 16-qubit Hamiltonian on the transpiled circuit while
   ignoring `circuit.layout`.

The net effect was the Hamiltonian-cycle ordering, by accident. In Qiskit 2 the idiom is
explicit and backend-agnostic:

```python
pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=...)
isa_circuit = pm.run(circuit)
isa_observable = observable.apply_layout(isa_circuit.layout)
```

A naive port of the notebook with `apply_layout` but identity site order would silently make
the benchmark unsolvable by the published ansatzes. Hence `site_order` is a first-class,
tested concept here (`transpile.py`, `tests/unit/test_transpile_estimators.py`).

For Guadalupe, `GUADALUPE_HEAVY_HEX_LOOP = (1, 2, 3, 5, 8, 11, 14, 13, 12, 10, 7, 4)` is the
recommended `initial_layout`: ring gates then need no routing SWAPs. Kagome contains
triangles and heavy-hex does not, so the six hexagon bonds can never all be physical
couplings; that only matters for gates, not for measuring the observable.

## 6. eSWAP

The eSWAP gate is `exp(-i phi/2 SWAP)`. Since `SWAP = (I + XX + YY + ZZ)/2` and the three
Pauli products commute,

```
exp(-i phi/2 SWAP) = e^{-i phi/4} RXX(phi/2) RYY(phi/2) RZZ(phi/2)
```

which is how `eswap()` is built. `eswap(0)` is the identity and `eswap(pi)` is SWAP up to a
global phase, so a ring of eSWAPs interpolates between dimer coverings: the resonating valence
bond picture.

The original notebook's gate (`legacy_eswap`) is `CX(1,0) CRX(phi; 0->1) X(0) RZ(-phi/2, 0)
CX(1,0)`, which equals `X(0) . exp(-i phi/2 SWAP)` up to a global phase. The stray `X`
means `legacy_eswap(0)` is not the identity. On the singlet product this happens to be
harmless in noiseless simulation, but it is a different circuit on hardware. It is kept for
reproduction only (`ansatz.gate = "legacy"`).

## 7. Lattice-aware HVA

`HVAnsatz(extra_bonds=...)`, enabled by `ansatz.lattice_aware = true`, adds a third sub-layer
with its own parameter on the lattice bonds that are not on the ring: the six hexagon bonds.
The ansatz then "sees" the whole Hamiltonian at the cost of extra two-qubit depth (and of
routing, since those bonds are not heavy-hex couplings).

## 8. What the benchmark measures

With the default site order the exact ground state sits at all-zero parameters. A noiseless
run therefore mostly tests whether the optimiser walks back from the random start
`x0 = pi/4 * rand`; HVA does so easily (2 parameters), RVB's SPSA run lands at 0.9 %.

The physically meaningful benchmark is the noisy one. On `aer_noisy:FakeGuadalupeV2` the
transpiled circuits have 66 (HVA) to 93 (RVB) CX gates and the energy at the exact optimum is
already far above -18. The questions worth asking there are: how much does resilience level
1 or 2 recover, which ansatz survives the depth, and how large a perturbation SPSA needs to
see through shot noise. Those are exactly the knobs `BackendConfig` exposes.

## 9. Extensions

* **Larger / periodic kagome.** `Lattice` is a plain edge list; a `kagome_lattice(rows, cols,
  periodic=True)` constructor plus a Hamiltonian-cycle (or matching-based) `site_order` is the
  natural next step. Exact diagonalisation stays feasible to roughly 20 sites.
* **Parameter-shift gradients** apply gate by gate. Each eSWAP is `exp(-i phi/2 SWAP)` and
  `SWAP/2` has eigenvalues `+-1/2`, so the two-term rule gives RVB's exact gradient from 24
  energies. In HVA each parameter is shared by 18 rotations: the rule applies per occurrence
  and the occurrences are summed (72 energies), because the summed generator has many
  eigenvalues. Only the legacy eSWAP, with its controlled rotation, needs a longer rule. The
  package currently implements finite differences only (`fd_eps`, raised to ~0.1 under shot
  noise); see `docs/roadmap.md` for the cost and noise comparison.
* **Error mitigation studies.** `resilience_level`, `shots` and `initial_layout` are per-run
  config; a benchmark config with several `[[combos]]` can sweep them.
