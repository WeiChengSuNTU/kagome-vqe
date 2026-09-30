"""Shared fixtures."""

from __future__ import annotations

import os

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from kagome_vqe.hamiltonian import heisenberg_hamiltonian
from kagome_vqe.lattice import Lattice, kagome_unit_cell


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``ibm`` tests unless explicitly enabled; they cost real device time."""
    if os.environ.get("KAGOME_RUN_IBM") == "1":
        return
    skip = pytest.mark.skip(reason="set KAGOME_RUN_IBM=1 to run tests against IBM Quantum")
    for item in items:
        if "ibm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def lattice() -> Lattice:
    """Kagome cell with the default Hamiltonian-cycle site order."""
    return kagome_unit_cell()


@pytest.fixture(scope="session")
def identity_lattice() -> Lattice:
    """Kagome cell with identity site order (the hard variant)."""
    return kagome_unit_cell(site_order=None)


@pytest.fixture(scope="session")
def hamiltonian(lattice: Lattice) -> SparsePauliOp:
    """Hamiltonian in the default order."""
    return heisenberg_hamiltonian(lattice)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator."""
    return np.random.default_rng(1234)
