# Hardware mapping benchmarks (compile only)

Resource counts of the `graph_hva` circuit mapped onto IBM devices with the three mapping
strategies. Nothing is executed: `kagome-vqe map` compiles against the device `Target`
(a fake backend, or with `-b ibm:<device>` the real device's target, fetched without
submitting a job). The strategies themselves are described in [`graph_hva.md`](graph_hva.md);
credentials and execution on IBM Quantum in [`ibm.md`](ibm.md).

## Reproduction

```bash
uv run kagome-vqe map configs/examples/07_nighthawk_compile_only.toml               # FakeNighthawk
uv run kagome-vqe map configs/examples/07_nighthawk_compile_only.toml -b ibm:ibm_phoenix   # real Target
uv run kagome-vqe map configs/examples/05_mapping_prb106.toml --out results/map-prb106
```

## Results (2026-11-10)

**Hardware mapping, compile only** (`kagome-vqe map`, `graph_hva` with `p = 2`, no H0,
optimisation level 3, seed 11; SWAPs for `generic` are those SABRE inserts at level 1;
"block 2q depth" counts exchange and SWAP gates as one layer each, the papers' convention):

PRB 106 20-site patch (`schedule = "prb106"`) on FakeNighthawk:

| strategy | status | physical qubits | auxiliary | SWAPs | block 2q depth | CZ gates | CZ depth | total depth |
|---|---|---|---|---|---|---|---|---|
| prb105 | paper_inspired | 24 | 4 | 40 | 19 | 295 | 55 | 210 |
| prb106_grid | paper_exact | 24 | 4 | 32 | 15 | 286 | 43 | 183 |
| generic | generic | 20 | 0 | 29 | - | 237 | 63 | 259 |

PRB 105 3x8 rectangle (`schedule = "prb105"`) on FakeNighthawk:

| strategy | status | physical qubits | auxiliary | SWAPs | block 2q depth | CZ gates | CZ depth | total depth |
|---|---|---|---|---|---|---|---|---|
| prb105 | paper_exact | 24 | 0 | 28 | 15 | 292 | 43 | 174 |
| prb106_grid | paper_inspired | 33 | 9 | 52 | 19 | 396 | 55 | 189 |
| generic | generic | 25 | 0 | 38 | - | 318 | 76 | 287 |

12-site star (`schedule = "greedy"`) on FakeNighthawk:

| strategy | status | physical qubits | auxiliary | SWAPs | block 2q depth | CZ gates | CZ depth | total depth |
|---|---|---|---|---|---|---|---|---|
| prb105 | paper_inspired | 13 | 1 | 22 | 24 | 171 | 62 | 228 |
| prb106_grid | paper_inspired | 17 | 5 | 26 | 17 | 192 | 49 | 172 |
| generic | generic | 12 | 0 | 17 | - | 136 | 35 | 147 |

PRB 106 20-site patch on FakeBoston (Heron, heavy-hex): `prb105` and `prb106_grid` are reported
`unsupported`, because the routed grid does not embed in the coupling map. `generic` needs
22 qubits, 66 SWAPs and 327 CZ gates at CZ depth 93.

Each paper's own construction is the shallowest option on its own lattice. On the 20-site
patch, `prb106_grid` gives CZ depth 43 against 63 for generic routing; on the 3x8 rectangle,
`prb105` gives 43 against 76. Generic routing still uses fewer CZ gates on the 20-site patch
(237 against 286), because it needs no stations. On the small star, generic routing wins
outright. None of the strategies is better everywhere; which one is best depends on the
lattice, the device and whether depth or gate count limits fidelity.
