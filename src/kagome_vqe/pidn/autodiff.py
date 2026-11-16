"""A small reverse-mode automatic differentiation engine on numpy, with higher-order gradients.

PIDN's physics-informed loss compares the *input gradient* of the network,
``grad_theta C_hat(theta_t)``, with the zero-noise-extrapolated gradient ``g_ZNE``, and training
needs the derivative of that loss with respect to the weights: a gradient of a gradient
("double backpropagation"). This module provides exactly that, without a deep-learning
framework: every backward rule is itself written with differentiable :class:`Tensor`
operations, so :func:`grad` with ``create_graph=True`` returns tensors that can be
differentiated again.

The engine is deliberately minimal: float64 arrays, the operations an MLP and a GRU need
(``matmul``, elementwise arithmetic with broadcasting, ``tanh``, ``sigmoid``, ``relu``,
``sin``/``cos``, reductions, reshapes, slicing, concatenation) and nothing else. Every rule is
checked against finite differences in ``tests/unit/test_pidn.py``, to second order.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Tensor",
    "as_tensor",
    "concat",
    "cos",
    "grad",
    "matmul",
    "no_grad",
    "relu",
    "sigmoid",
    "silu",
    "sin",
    "tanh",
]

FloatArray = NDArray[np.float64]
Backward = Callable[["Tensor"], tuple["Tensor | None", ...]]

_RECORDING = [True]


class no_grad:  # noqa: N801 - mirrors the familiar context-manager name
    """Context manager: operations inside build no graph (inference, plain backward)."""

    def __enter__(self) -> None:
        """Stop recording."""
        self._previous = _RECORDING[0]
        _RECORDING[0] = False

    def __exit__(self, *exc: object) -> None:
        """Restore the previous recording state."""
        _RECORDING[0] = self._previous


class Tensor:
    """An array that remembers how it was computed.

    Attributes:
        value: The numpy array (float64).
        requires_grad: Whether gradients flow to (or through) this tensor.
    """

    __slots__ = ("_backward", "_parents", "requires_grad", "value")

    def __init__(
        self,
        value: Any,
        *,
        requires_grad: bool = False,
        parents: tuple[Tensor, ...] = (),
        backward: Backward | None = None,
    ) -> None:
        """Wrap ``value``; graph fields are set by the operations."""
        self.value: FloatArray = np.asarray(value, dtype=float)
        self.requires_grad = requires_grad
        self._parents = parents
        self._backward = backward

    # -- construction helpers ----------------------------------------------------------------

    @staticmethod
    def _make(value: Any, parents: tuple[Tensor, ...], backward: Backward) -> Tensor:
        needs = _RECORDING[0] and any(p.requires_grad for p in parents)
        if not needs:
            return Tensor(value)
        return Tensor(value, requires_grad=True, parents=parents, backward=backward)

    def detach(self) -> Tensor:
        """The same values, cut from the graph."""
        return Tensor(self.value.copy())

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        return tuple(self.value.shape)

    def numpy(self) -> FloatArray:
        """A copy of the values."""
        return self.value.copy()

    def __repr__(self) -> str:
        """Debug representation."""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # -- arithmetic --------------------------------------------------------------------------

    def __add__(self, other: Any) -> Tensor:
        """Elementwise sum with broadcasting."""
        b = as_tensor(other)
        a = self

        def backward(g: Tensor) -> tuple[Tensor | None, ...]:
            return g.sum_to(a.shape), g.sum_to(b.shape)

        return Tensor._make(a.value + b.value, (a, b), backward)

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        """Elementwise negation."""
        a = self
        return Tensor._make(-a.value, (a,), lambda g: (-g,))

    def __sub__(self, other: Any) -> Tensor:
        """Elementwise difference with broadcasting."""
        return self + (-as_tensor(other))

    def __rsub__(self, other: Any) -> Tensor:
        """``other - self``."""
        return as_tensor(other) + (-self)

    def __mul__(self, other: Any) -> Tensor:
        """Elementwise product with broadcasting."""
        b = as_tensor(other)
        a = self

        def backward(g: Tensor) -> tuple[Tensor | None, ...]:
            return (g * b).sum_to(a.shape), (g * a).sum_to(b.shape)

        return Tensor._make(a.value * b.value, (a, b), backward)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Tensor:
        """Division by a constant (tensors in the denominator are not needed here)."""
        b = as_tensor(other)
        if b.requires_grad:
            msg = "division by a tensor that requires grad is not supported"
            raise NotImplementedError(msg)
        return self * Tensor(1.0 / b.value)

    def __matmul__(self, other: Any) -> Tensor:
        """Matrix product of 2-D tensors."""
        return matmul(self, as_tensor(other))

    def square(self) -> Tensor:
        """Elementwise square."""
        return self * self

    # -- reductions and shapes ---------------------------------------------------------------

    def sum(self, axis: int | None = None, *, keepdims: bool = False) -> Tensor:
        """Sum over ``axis`` (all axes if ``None``)."""
        a = self
        value = np.asarray(
            np.sum(a.value, axis=axis, keepdims=True) if keepdims else np.sum(a.value, axis=axis)
        )

        def backward(g: Tensor) -> tuple[Tensor | None, ...]:
            if axis is not None and not keepdims:
                g = g.reshape(tuple(np.expand_dims(g.value, axis).shape))
            return (g.broadcast_to(a.shape),)

        return Tensor._make(value, (a,), backward)

    def mean(self, axis: int | None = None, *, keepdims: bool = False) -> Tensor:
        """Mean over ``axis``."""
        count = self.value.size if axis is None else self.value.shape[axis]
        return self.sum(axis, keepdims=keepdims) * (1.0 / count)

    def reshape(self, shape: Sequence[int]) -> Tensor:
        """Reshape."""
        a = self
        target = tuple(shape)
        return Tensor._make(a.value.reshape(target), (a,), lambda g: (g.reshape(a.shape),))

    def transpose(self) -> Tensor:
        """Transpose of a 2-D tensor."""
        a = self
        return Tensor._make(a.value.T, (a,), lambda g: (g.transpose(),))

    @property
    def T(self) -> Tensor:  # noqa: N802 - numpy convention
        """Transpose of a 2-D tensor."""
        return self.transpose()

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        """Broadcast to ``shape``; the backward sums back."""
        a = self
        if a.shape == tuple(shape):
            return a
        return Tensor._make(
            np.broadcast_to(a.value, shape).copy(), (a,), lambda g: (g.sum_to(a.shape),)
        )

    def sum_to(self, shape: tuple[int, ...]) -> Tensor:
        """Sum broadcast axes away so the result has ``shape`` (inverse of broadcasting)."""
        a = self
        if a.shape == tuple(shape):
            return a
        value = _sum_to(a.value, shape)
        return Tensor._make(value, (a,), lambda g: (g.broadcast_to(a.shape),))

    def __getitem__(self, key: Any) -> Tensor:
        """Basic slicing/indexing; the backward scatters into zeros."""
        a = self
        value = np.asarray(a.value[key])

        def backward(g: Tensor) -> tuple[Tensor | None, ...]:
            return (_scatter(g, key, a.shape),)

        return Tensor._make(value, (a,), backward)


# -- free functions -----------------------------------------------------------------------------


def as_tensor(value: Any) -> Tensor:
    """``value`` if it already is a tensor, else a constant tensor."""
    return value if isinstance(value, Tensor) else Tensor(value)


def _sum_to(value: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    shape = tuple(shape)
    extra = value.ndim - len(shape)
    out = value.sum(axis=tuple(range(extra))) if extra > 0 else value
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and out.shape[i] != 1)
    if axes:
        out = out.sum(axis=axes, keepdims=True)
    return np.asarray(out.reshape(shape), dtype=float)


def _scatter(g: Tensor, key: Any, shape: tuple[int, ...]) -> Tensor:
    """Zeros of ``shape`` with ``g`` placed at ``key`` (differentiable in ``g``)."""
    out = np.zeros(shape)
    out[key] = g.value
    return Tensor._make(out, (g,), lambda h: (h[key],))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` for 2-D tensors."""

    def backward(g: Tensor) -> tuple[Tensor | None, ...]:
        return matmul(g, b.transpose()), matmul(a.transpose(), g)

    return Tensor._make(a.value @ b.value, (a, b), backward)


def tanh(x: Tensor) -> Tensor:
    """Elementwise ``tanh``."""
    y_value = np.tanh(x.value)
    out: Tensor

    def backward(g: Tensor) -> tuple[Tensor | None, ...]:
        return (g * (1.0 - out * out),)

    out = Tensor._make(y_value, (x,), backward)
    return out


def sigmoid(x: Tensor) -> Tensor:
    """Elementwise logistic function."""
    y_value = 0.5 * (1.0 + np.tanh(0.5 * x.value))
    out: Tensor

    def backward(g: Tensor) -> tuple[Tensor | None, ...]:
        return (g * out * (1.0 - out),)

    out = Tensor._make(y_value, (x,), backward)
    return out


def relu(x: Tensor) -> Tensor:
    """Elementwise ``max(x, 0)``; its second derivative is zero almost everywhere."""
    mask = (x.value > 0).astype(float)
    return Tensor._make(x.value * mask, (x,), lambda g: (g * Tensor(mask),))


def silu(x: Tensor) -> Tensor:
    """``x sigmoid(x)`` (smooth, ReLU-like)."""
    return x * sigmoid(x)


def sin(x: Tensor) -> Tensor:
    """Elementwise sine."""
    return Tensor._make(np.sin(x.value), (x,), lambda g: (g * cos(x),))


def cos(x: Tensor) -> Tensor:
    """Elementwise cosine."""
    return Tensor._make(np.cos(x.value), (x,), lambda g: (-(g * sin(x)),))


def concat(tensors: Sequence[Tensor], axis: int = -1) -> Tensor:
    """Concatenate along ``axis``."""
    parts = [as_tensor(t) for t in tensors]
    value = np.concatenate([t.value for t in parts], axis=axis)
    ax = axis % value.ndim
    bounds = np.cumsum([0] + [t.value.shape[ax] for t in parts])

    def backward(g: Tensor) -> tuple[Tensor | None, ...]:
        out = []
        for lo, hi in itertools.pairwise(bounds):
            key = tuple([slice(None)] * ax + [slice(int(lo), int(hi))])
            out.append(g[key])
        return tuple(out)

    return Tensor._make(value, tuple(parts), backward)


def _topological(root: Tensor) -> list[Tensor]:
    order: list[Tensor] = []
    seen: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        stack.extend(
            (parent, False)
            for parent in node._parents
            if parent.requires_grad and id(parent) not in seen
        )
    return order


def grad(
    output: Tensor,
    inputs: Iterable[Tensor],
    *,
    create_graph: bool = False,
    seed: Tensor | None = None,
) -> list[Tensor]:
    """Gradients of a scalar ``output`` (or of ``sum(seed * output)``) w.r.t. ``inputs``.

    With ``create_graph=True`` the returned tensors are part of the graph and can be
    differentiated again; otherwise they are constants.
    """
    inputs = list(inputs)
    if seed is None:
        if output.value.size != 1:
            msg = "grad of a non-scalar output needs a seed"
            raise ValueError(msg)
        seed = Tensor(np.ones_like(output.value))
    previous = _RECORDING[0]
    _RECORDING[0] = create_graph
    try:
        grads: dict[int, Tensor] = {id(output): seed}
        for node in reversed(_topological(output)):
            g = grads.get(id(node))
            if g is None or node._backward is None:
                continue
            for parent, pg in zip(node._parents, node._backward(g), strict=True):
                if pg is None or not parent.requires_grad:
                    continue
                key = id(parent)
                grads[key] = pg if key not in grads else grads[key] + pg
    finally:
        _RECORDING[0] = previous
    out = []
    for x in inputs:
        g = grads.get(id(x))
        if g is None:
            g = Tensor(np.zeros_like(x.value))
        out.append(g if create_graph else Tensor(g.value))
    return out
