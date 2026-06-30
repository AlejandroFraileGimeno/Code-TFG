# -*- coding: utf-8 -*-
"""
Heatmap de transmitancia total vs espesor y número de onda
Estructura: Air / material(d, phi=0) / sustrato
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
MATERIAL    = MoO3        # Material de la capa: MoO3, V2O5, MgTeMoO6
SUBSTRATE   = BaF2        # Sustrato: Au, SiO2, BaF2, Air
ALPHA       = 0.0         # Ángulo de incidencia (radianes). 0 = normal
D_MIN       = 50         # Espesor mínimo (nm)
D_MAX       = 1000        # Espesor máximo (nm)
N_THICKNESS = 1000         # Número de puntos de espesor
FREQ_MIN    = 500         # Frecuencia mínima (cm⁻¹)
FREQ_MAX    = 1100        # Frecuencia máxima (cm⁻¹)
N_FREQS     = 500         # Número de puntos de frecuencia
T_COMPONENT = "ss"        # Componente: "pp", "ss", "ps", "sp"
# ---------------------------------------------------------------------------

_T_INDEX = {"pp": 0, "ss": 1, "ps": 2, "sp": 3}
assert T_COMPONENT in _T_INDEX, f"T_COMPONENT debe ser uno de {list(_T_INDEX)}"

thicknesses = np.linspace(D_MIN, D_MAX, N_THICKNESS)
freqs       = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
T_map       = np.zeros((N_THICKNESS, N_FREQS))

print(f"Calculando heatmap: {N_THICKNESS} espesores × {N_FREQS} frecuencias...")
for i, d in enumerate(thicknesses):
    if i % max(1, N_THICKNESS // 10) == 0:
        print(f"  {100 * i / N_THICKNESS:.0f}%")
    structure = LayeredStructure(
        superstrate=Air(),
        substrate=SUBSTRATE(),
        layers=[MATERIAL(d=d * 1e-9, phi=0.0)],
    )
    for j, f in enumerate(freqs):
        T_map[i, j] = calculate_transmission(f, ALPHA, structure, basis="linear")[_T_INDEX[T_COMPONENT]]

print("  100% — listo")

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.pcolormesh(freqs, thicknesses, T_map, cmap="viridis", shading="auto")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"$T_{{{T_COMPONENT}}}$")

ax.set_xlabel("Número de onda (cm⁻¹)", fontsize=14)
ax.set_ylabel("Espesor (nm)", fontsize=14)
fig.tight_layout()
plt.show()
