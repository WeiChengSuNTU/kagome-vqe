"""Hardware mapping: embed a logical circuit into physical connectivity.

The logical ansatz is built from the lattice alone (:mod:`kagome_vqe.ansatz`). A mapping
strategy then decides where each logical qubit lives and which SWAPs implement the bonds that
are not hardware couplers:

* :func:`map_circuit` with ``"prb105"`` / ``"prb106_grid"`` / ``"generic"``
  (:mod:`kagome_vqe.mapping.strategies`);
* :func:`place_on_backend` embeds the routed grid into a backend's coupling map
  (:mod:`kagome_vqe.mapping.placement`).
"""

from kagome_vqe.mapping.core import (
    LogicalOp,
    MappingError,
    MoveRules,
    RoutedCircuit,
    SwapRouter,
    exchange_cx3,
    logical_ops,
    routed_observable,
    synthesise,
    verify_routed,
)
from kagome_vqe.mapping.strategies import (
    MAPPING_STRATEGIES,
    MappingResult,
    MappingStatus,
    MappingStrategyName,
    block_two_qubit_depth,
    map_circuit,
    presynthesise_exchange,
)

__all__ = [
    "MAPPING_STRATEGIES",
    "LogicalOp",
    "MappingError",
    "MappingResult",
    "MappingStatus",
    "MappingStrategyName",
    "MoveRules",
    "RoutedCircuit",
    "SwapRouter",
    "block_two_qubit_depth",
    "exchange_cx3",
    "logical_ops",
    "map_circuit",
    "presynthesise_exchange",
    "routed_observable",
    "synthesise",
    "verify_routed",
]
