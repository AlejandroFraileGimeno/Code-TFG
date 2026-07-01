from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np

MatrixFn = Callable[[float], np.ndarray]


def _identity(_: float) -> np.ndarray:
    return np.eye(3, dtype=complex)


def _zeros(_: float) -> np.ndarray:
    return np.zeros((3, 3), dtype=complex)


@dataclass
class Layer:
    eps: MatrixFn = _identity
    mu: MatrixFn = _identity
    xi: MatrixFn = _zeros
    zeta: MatrixFn = _zeros
    d: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    psi: float = 0.0


@dataclass
class LayeredStructure:
    superstrate: Layer
    substrate: Layer
    layers: Sequence[Layer] = field(default_factory=list)
