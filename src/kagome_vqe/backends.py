"""Resolve backend names to Qiskit backend objects (fake or real).

Fake backends are discovered by introspecting ``qiskit_ibm_runtime.fake_provider`` at runtime
rather than hard-coding a list, because IBM adds and retires them with every release.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import cache

from qiskit.providers import BackendV2

from kagome_vqe.config import BackendConfig, Settings

__all__ = [
    "FakeBackendInfo",
    "fake_backend_names",
    "ibm_service",
    "list_fake_backends",
    "resolve_backend",
    "resolve_fake_backend",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FakeBackendInfo:
    """Summary of one fake backend."""

    name: str
    """Class name, e.g. ``"FakeGuadalupeV2"``."""
    backend_name: str
    """Device name reported by the backend, e.g. ``"fake_guadalupe"``."""
    num_qubits: int


@cache
def _fake_classes() -> dict[str, type[BackendV2]]:
    from qiskit_ibm_runtime import fake_provider
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

    return {
        name: obj
        for name, obj in inspect.getmembers(fake_provider, inspect.isclass)
        if name.startswith("Fake") and issubclass(obj, FakeBackendV2) and obj is not FakeBackendV2
    }


def _normalise(name: str) -> str:
    """Map ``"FakeGuadalupeV2"``, ``"fake_guadalupe"`` and ``"guadalupe"`` to ``"guadalupe"``."""
    return name.lower().replace("_", "").removeprefix("fake").removesuffix("v2")


def fake_backend_names() -> list[str]:
    """All fake backend class names, sorted."""
    return sorted(_fake_classes())


def resolve_fake_backend(name: str) -> BackendV2:
    """Instantiate the fake backend called ``name`` (``"FakeGuadalupeV2"`` or ``"fake_guadalupe"``).

    Raises:
        KeyError: If no such fake backend exists in the installed qiskit-ibm-runtime.
    """
    classes = _fake_classes()
    if name in classes:
        return classes[name]()
    wanted = _normalise(name)
    exact = [cls for cls_name, cls in classes.items() if _normalise(cls_name) == wanted]
    if len(exact) == 1:
        return exact[0]()
    msg = f"unknown fake backend {name!r}; run `kagome-vqe backends --fake` to list them"
    raise KeyError(msg)


def list_fake_backends(min_qubits: int = 0) -> list[FakeBackendInfo]:
    """Instantiate every fake backend and report those with at least ``min_qubits`` qubits.

    Instantiation loads calibration JSON for each device, so this takes a few seconds.
    """
    infos: list[FakeBackendInfo] = []
    for cls_name, cls in _fake_classes().items():
        try:
            backend = cls()
        except Exception as exc:
            log.debug("could not instantiate %s: %s", cls_name, exc)
            continue
        if backend.num_qubits >= min_qubits:
            infos.append(FakeBackendInfo(cls_name, str(backend.name), int(backend.num_qubits)))
    return sorted(infos, key=lambda i: (i.num_qubits, i.name))


def ibm_service(settings: Settings | None = None) -> object:
    """Create a ``QiskitRuntimeService`` from :class:`~kagome_vqe.config.Settings`.

    Falls back to credentials previously saved with ``QiskitRuntimeService.save_account`` when
    no token is present in the environment.

    Returns:
        The service (typed as ``object`` to keep qiskit-ibm-runtime an optional import here).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    settings = settings or Settings()
    kwargs: dict[str, object] = {"channel": settings.channel}
    if settings.has_credentials and settings.token is not None:
        kwargs["token"] = settings.token.get_secret_value()
    if settings.instance:
        kwargs["instance"] = settings.instance
    if settings.region:
        kwargs["region"] = settings.region
    log.info("connecting to IBM Quantum (channel=%s)", settings.channel)
    return QiskitRuntimeService(**kwargs)


def resolve_backend(cfg: BackendConfig, settings: Settings | None = None) -> BackendV2 | None:
    """Return the backend object a config refers to, or ``None`` for statevector.

    * ``statevector`` -> ``None`` (evaluated on the logical circuit).
    * ``aer_ideal`` -> an ideal ``AerSimulator`` (used as a transpilation target).
    * ``aer_noisy`` / ``fake`` -> the fake backend ``cfg.name``.
    * ``ibm`` -> the real device ``cfg.name`` from the runtime service.
    """
    if cfg.kind == "statevector":
        return None
    if cfg.kind == "aer_ideal":
        from qiskit_aer import AerSimulator

        return AerSimulator()
    assert cfg.name is not None  # guaranteed by BackendConfig validation
    if cfg.kind in ("aer_noisy", "fake"):
        return resolve_fake_backend(cfg.name)
    service = ibm_service(settings)
    backend: BackendV2 = service.backend(cfg.name)  # type: ignore[attr-defined]
    return backend
