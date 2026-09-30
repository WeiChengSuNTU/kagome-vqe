"""VQE for the spin-1/2 Heisenberg antiferromagnet on the kagome lattice.

The package is organised as a pipeline whose stages map one-to-one onto modules:

1. :mod:`kagome_vqe.lattice` -- the graph of spins (which sites, which bonds).
2. :mod:`kagome_vqe.hamiltonian` -- the graph turned into a ``SparsePauliOp`` plus exact
   diagonalisation for reference energies.
3. :mod:`kagome_vqe.ansatz` -- parameterised trial circuits (RVB, HVA, library ansatzes).
4. :mod:`kagome_vqe.transpile` -- compile a logical circuit + observable for a backend.
5. :mod:`kagome_vqe.estimators` -- evaluate energies on simulators, fake backends or IBM Quantum.
6. :mod:`kagome_vqe.optimizers` -- classical minimisers (ADAM, SPSA, scipy).
7. :mod:`kagome_vqe.vqe` -- the loop that ties 3-6 together and records a trace.
8. :mod:`kagome_vqe.results` / :mod:`kagome_vqe.plotting` -- persistence and figures.

Everything is driven by a :class:`kagome_vqe.config.RunConfig` (TOML on disk) and exposed
through the ``kagome-vqe`` command line (:mod:`kagome_vqe.cli`).
"""

from importlib.metadata import PackageNotFoundError, version

from kagome_vqe.hamiltonian import exact_ground_energy, exact_spectrum, heisenberg_hamiltonian
from kagome_vqe.lattice import Lattice, kagome_unit_cell

try:
    __version__ = version("kagome-vqe")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "Lattice",
    "__version__",
    "exact_ground_energy",
    "exact_spectrum",
    "heisenberg_hamiltonian",
    "kagome_unit_cell",
]
