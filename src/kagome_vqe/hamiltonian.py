"""Heisenberg Hamiltonians and exact reference energies.

The antiferromagnetic XXX Heisenberg model on a lattice with bonds ``<ij>`` and couplings
``J_ij`` is

    H = sum_<ij> J_ij (X_i X_j + Y_i Y_j + Z_i Z_j).

Built as a :class:`~qiskit.quantum_info.SparsePauliOp` directly from the bond list, in
ansatz-qubit labels (``Lattice.site_order`` applied). This replaces the qiskit-nature
``SpinOp`` + ``LogarithmicMapper`` construction of the original notebook, which needed a factor
of 4 to undo the spin-1/2 normalisation and no longer exists in Qiskit Nature 0.7+.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla
from qiskit.quantum_info import SparsePauliOp

from kagome_vqe.lattice import Lattice

_DENSE_MAX_DIM = 2**10
"""Up to 10 qubits the ground space is computed by dense diagonalisation (exact, fast)."""

__all__ = [
    "exact_ground_energy",
    "exact_ground_space",
    "exact_spectrum",
    "ground_state_fidelity",
    "heisenberg_hamiltonian",
    "relative_error",
]


def heisenberg_hamiltonian(lattice: Lattice) -> SparsePauliOp:
    """Build ``sum_<ij> J_ij (XX + YY + ZZ)`` on ``lattice.num_sites`` qubits.

    Qubit ``k`` hosts lattice site ``lattice.site_order[k]``. Terms are emitted bond by bond
    (XX, YY, ZZ per bond) and left unsimplified, so ``len(op)`` equals ``3 * num_bonds``.

    Args:
        lattice: The spin graph.

    Returns:
        A Hermitian operator with real coefficients.
    """
    terms: list[tuple[str, list[int], float]] = [
        (pauli, [a, b], w) for a, b, w in lattice.qubit_edges() for pauli in ("XX", "YY", "ZZ")
    ]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=lattice.num_sites)


def exact_spectrum(hamiltonian: SparsePauliOp, k: int = 3) -> np.ndarray:
    """Lowest ``k`` eigenvalues by sparse Lanczos (``scipy.sparse.linalg.eigsh``).

    The 12-qubit kagome Hamiltonian is a 4096 x 4096 matrix with ~78k non-zeros; this takes
    well under a second. Fine up to roughly 20 qubits; beyond that use a dedicated ED code.

    Args:
        hamiltonian: The operator (must be Hermitian).
        k: How many eigenvalues to return, ascending.

    Returns:
        Real eigenvalues, ascending, shape ``(k,)``.
    """
    matrix = hamiltonian.to_matrix(sparse=True)
    dim = matrix.shape[0]
    if k >= dim - 1:  # eigsh needs k < dim - 1; tiny systems fall back to dense
        dense = np.linalg.eigvalsh(matrix.toarray())
        return np.asarray(dense[:k], dtype=float)
    values = spla.eigsh(matrix, k=k, which="SA", return_eigenvectors=False)
    return np.sort(np.real(values))


def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """The lowest eigenvalue of ``hamiltonian``."""
    return float(exact_spectrum(hamiltonian, k=1)[0])


def relative_error(target: float, value: float) -> float:
    """``|target - value| / |target|``; the IBM challenge's figure of merit (must be < 1 %)."""
    if target == 0:
        msg = "relative error is undefined for a zero target"
        raise ZeroDivisionError(msg)
    return abs((target - value) / target)


def exact_ground_space(
    hamiltonian: SparsePauliOp, *, k: int = 6, tol: float = 1e-8
) -> tuple[float, np.ndarray]:
    """Ground energy and an orthonormal basis of the (possibly degenerate) ground space.

    Args:
        hamiltonian: The operator.
        k: Eigenpairs to compute; the ground space is assumed to be at most ``k - 1`` fold
            degenerate (a warning-free check: if all ``k`` are degenerate, ``k`` is too small).
        tol: Energies within ``tol`` of the minimum count as degenerate.

    Returns:
        ``(E0, V)`` with ``V`` of shape ``(2**n, d)``.

    Raises:
        ValueError: If all ``k`` computed eigenvalues are degenerate.
    """
    matrix = hamiltonian.to_matrix(sparse=True)
    dim = matrix.shape[0]
    if dim <= _DENSE_MAX_DIM:
        values, vectors = np.linalg.eigh(matrix.toarray())
        values, vectors = values[:k], vectors[:, :k]
    else:
        values, vectors = spla.eigsh(matrix, k=k, which="SA", tol=1e-12, ncv=max(4 * k, 40))
    order = np.argsort(values)
    values, vectors = np.real(values[order]), vectors[:, order]
    degenerate = values - values[0] < tol
    if degenerate.all() and k < dim:
        msg = f"all {k} lowest eigenvalues are degenerate; increase k"
        raise ValueError(msg)
    # Lanczos returns degenerate eigenvectors that are only approximately orthogonal (overlaps
    # ~1e-3 were observed), which would make the projector, and fidelities, wrong.
    basis, _ = np.linalg.qr(vectors[:, degenerate])
    residual = np.linalg.norm(matrix @ basis - values[0] * basis, axis=0).max()
    if residual > 1e-6:
        msg = f"ground space not converged (residual {residual:.1e})"
        raise ValueError(msg)
    return float(values[0]), basis


def ground_state_fidelity(state: np.ndarray, ground_space: np.ndarray) -> float:
    """``sum_k |<E0_k|psi>|^2``: overlap of ``state`` with the (degenerate) ground space."""
    overlaps = ground_space.conj().T @ np.asarray(state)
    return float(np.sum(np.abs(overlaps) ** 2))
