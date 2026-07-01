# -*- coding: utf-8 -*-
"""
Mapa de transmitancia vs ángulo phi y número de onda
Estructura: Air / material(d, phi) / sustrato
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
THICKNESS   = 100e-9      # Espesor fijo (metros)
ALPHA       = 0.0         # Ángulo de incidencia (radianes). 0 = normal
PHI_MIN     = 0           # Ángulo phi mínimo (grados)
PHI_MAX     = 90          # Ángulo phi máximo (grados)
N_PHI       = 200         # Número de puntos de phi
FREQ_MIN    = 450         # Frecuencia mínima (cm⁻¹)
FREQ_MAX    = 1100        # Frecuencia máxima (cm⁻¹)
N_FREQS     = 500         # Número de puntos de frecuencia
T_COMPONENT = "ss"        # Componente: "pp", "ss", "ps", "sp"
PLOT_TYPE   = "heatmap"   # "heatmap" o "3d"
AZIMUTH     = -60         # Ángulo azimutal para la vista 3D (grados)
ELEVATION   = 30          # Ángulo de elevación para la vista 3D (grados)
# ---------------------------------------------------------------------------

_T_INDEX = {"pp": 0, "ss": 1, "ps": 2, "sp": 3}
_T_LABEL = {"pp": "xx", "ss": "yy", "ps": "xy", "sp": "yx"}
assert T_COMPONENT in _T_INDEX, f"T_COMPONENT debe ser uno de {list(_T_INDEX)}"

_label = _T_LABEL[T_COMPONENT]

phis  = np.linspace(PHI_MIN, PHI_MAX, N_PHI)          # grados
freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
T_map = np.zeros((N_PHI, N_FREQS))

print(f"Calculando mapa: {N_PHI} ángulos × {N_FREQS} frecuencias...")
for i, phi_deg in enumerate(phis):
    if i % max(1, N_PHI // 10) == 0:
        print(f"  {100 * i / N_PHI:.0f}%")
    structure = LayeredStructure(
        superstrate=Air(),
        substrate=SUBSTRATE(),
        layers=[MATERIAL(d=THICKNESS, phi=phi_deg * np.pi / 180.0)],
    )
    for j, f in enumerate(freqs):
        T_map[i, j] = calculate_transmission(f, ALPHA, structure, basis="linear")[_T_INDEX[T_COMPONENT]]

print("  100% — listo")

FREQ_GRID, PHI_GRID = np.meshgrid(freqs, phis)

if PLOT_TYPE == "3d":
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        FREQ_GRID, PHI_GRID, T_map,
        cmap="viridis", linewidth=0, antialiased=True, alpha=0.95,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(f"$T_{{{_label}}}$", fontsize=16)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=12, labelpad=10)
    ax.set_ylabel(r"$\phi$ (°)", fontsize=12, labelpad=10)
    ax.set_zlabel(f"$T_{{{_label}}}$", fontsize=14, labelpad=8)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(freqs, phis, T_map, cmap="viridis", shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"$T_{{{_label}}}$", fontsize=18)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=14)
    ax.set_ylabel(r"$\phi$ (°)", fontsize=14)

fig.tight_layout()
plt.show()
