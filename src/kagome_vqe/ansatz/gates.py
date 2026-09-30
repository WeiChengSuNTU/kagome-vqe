"""Two-qubit building blocks shared by the ansatzes."""

from __future__ import annotations

from qiskit.circuit import Gate, ParameterExpression, QuantumCircuit

__all__ = ["eswap", "legacy_eswap", "singlet_pair"]


def singlet_pair(circuit: QuantumCircuit, a: int, b: int) -> None:
    """Prepare ``(|01> - |10>)/sqrt(2)`` on qubits ``(a, b)`` in place, from ``|00>``.

    Sequence ``X X, H(a), CX(a, b)``: after the X gates the pair is ``|11>``, ``H`` on ``a``
    gives ``(|0> - |1>)|1>/sqrt(2)`` and the CNOT flips ``b`` when ``a`` is 1, leaving the
    singlet. The singlet is the two-site ground state of the antiferromagnetic bond
    ``XX + YY + ZZ`` with energy ``-3``.
    """
    circuit.x(a)
    circuit.x(b)
    circuit.h(a)
    circuit.cx(a, b)


def eswap(phi: ParameterExpression | float) -> Gate:
    """The eSWAP gate ``exp(-i phi/2 SWAP)`` up to a global phase.

    Since ``SWAP = (I + XX + YY + ZZ)/2`` and the three Pauli products commute,
    ``exp(-i phi/2 SWAP) = e^{-i phi/4} RXX(phi/2) RYY(phi/2) RZZ(phi/2)``. ``eswap(0)`` is
    the identity and ``eswap(pi)`` is a SWAP (up to phase), so a ring of eSWAPs interpolates
    between fixed dimer coverings, which is the resonating-valence-bond picture.

    Args:
        phi: Rotation angle; a circuit parameter or a float.

    Returns:
        A gate named ``eswap`` that the transpiler decomposes into native gates.
    """
    sub = QuantumCircuit(2, name="eswap")
    sub.rxx(phi / 2, 0, 1)
    sub.ryy(phi / 2, 0, 1)
    sub.rzz(phi / 2, 0, 1)
    return sub.to_gate(label="eSWAP")


def legacy_eswap(phi: ParameterExpression | float) -> Gate:
    """The gate the original notebook called eSWAP, kept for reproduction.

    Verified equal to ``X(0) . exp(-i phi/2 SWAP)`` up to a global phase: the sequence
    contains a stray ``X`` on the first qubit, so ``legacy_eswap(0)`` is ``X`` on qubit 0, not
    the identity. Prefer :func:`eswap` for new work.
    """
    sub = QuantumCircuit(2, name="legacy_eswap")
    sub.cx(1, 0)
    sub.crx(phi, 0, 1)
    sub.x(0)
    sub.rz(-phi / 2, 0)
    sub.cx(1, 0)
    return sub.to_gate(label="eSWAP(legacy)")
