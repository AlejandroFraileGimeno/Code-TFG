# -*- coding: utf-8 -*-
"""
Transmitancia vs número de onda — estructura de 1 capa.
Estructura: Air / material(d, phi) / sustrato
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, Au, SiO2, BaF2,
    MoO3, V2O5, MgTeMoO6, hBN,
    LayeredStructure,
    calculate_transmission,
)

# ---------------------------------------------------------------------------
# CONFIG — modifica aquí
# ---------------------------------------------------------------------------
MATERIAL    = MoO3        # Material de la capa: MoO3, V2O5, MgTeMoO6, hBN
SUBSTRATE   = BaF2        # Sustrato: Au, SiO2, BaF2, Air
THICKNESS   = 200e-9     # Espesor en metros (ej: 1000e-9 = 1000 nm)
PHI         = 0.0         # Orientación del cristal (radianes)
ALPHA       = 0.0         # Ángulo de incidencia (radianes). 0 = normal
FREQ_MIN    = 500         # Frecuencia mínima (cm⁻¹)
FREQ_MAX    = 1100        # Frecuencia máxima (cm⁻¹)
N_FREQS     = 1000        # Número de puntos de frecuencia
T_COMPONENT = "pp"        # "pp", "ss", "ps", "sp"
# ---------------------------------------------------------------------------

_T_INDEX = {"pp": 0, "ss": 1, "ps": 2, "sp": 3}
_T_LABEL = {"pp": "xx", "ss": "yy", "ps": "xy", "sp": "yx"}
assert T_COMPONENT in _T_INDEX, f"T_COMPONENT debe ser uno de {list(_T_INDEX)}"

_label = _T_LABEL[T_COMPONENT]

omega = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

structure = LayeredStructure(
    superstrate=Air(),
    substrate=SUBSTRATE(),
    layers=[MATERIAL(d=THICKNESS, phi=PHI)],
)

T = np.array([
    calculate_transmission(f, ALPHA, structure, basis="linear")[_T_INDEX[T_COMPONENT]]
    for f in omega
])

# ---------------------------------------------------------------------------
# Estilo publicación
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
})

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(omega, T, color="black", linewidth=1.8, label=f"$T_{{{_label}}}$")

ax.set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$", fontsize=13)
ax.set_ylabel("Transmitancia", fontsize=13)
ax.set_xlim(FREQ_MIN, FREQ_MAX)
ax.set_ylim(0, 1)
ax.legend(fontsize=12, framealpha=0.9)

fig.tight_layout()
plt.show()
