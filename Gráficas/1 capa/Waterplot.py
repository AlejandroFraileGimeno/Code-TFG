# -*- coding: utf-8 -*-
"""
Mapa de transmitancia vs espesor y número de onda
Estructura: Air / material(d, phi=0) / sustrato
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
MATERIAL    = MoO3        # Material de la capa: MoO3, V2O5, MgTeMoO6
SUBSTRATE   = BaF2        # Sustrato: Au, SiO2, BaF2, Air
ALPHA       = 0.0         # Ángulo de incidencia (radianes). 0 = normal
D_MIN       = 50          # Espesor mínimo (nm)
D_MAX       = 1000        # Espesor máximo (nm)
N_THICKNESS = 1000         # Número de puntos de espesor
FREQ_MIN    = 450         # Frecuencia mínima (cm⁻¹)
FREQ_MAX    = 1100        # Frecuencia máxima (cm⁻¹)
N_FREQS     = 500         # Número de puntos de frecuencia
T_COMPONENT = "ss"        # Componente: "pp", "ss", "ps", "sp"
PLOT_TYPE   = "heatmap"   # "heatmap" o "3d"
AZIMUTH     = -60         # Ángulo azimutal para la vista 3D (grados)
ELEVATION   = 30          # Ángulo de elevación para la vista 3D (grados)
# ---------------------------------------------------------------------------

if len(sys.argv) > 1:  # permite: python Waterplot.py ss
    T_COMPONENT = sys.argv[1]

_T_INDEX = {"pp": 0, "ss": 1, "ps": 2, "sp": 3}
_T_LABEL = {"pp": "xx", "ss": "yy", "ps": "xy", "sp": "yx"}
assert T_COMPONENT in _T_INDEX, f"T_COMPONENT debe ser uno de {list(_T_INDEX)}"

_label = _T_LABEL[T_COMPONENT]

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
        layers=[MATERIAL(d=d * 1e-9, phi=0.0*np.pi/180.0)],  # phi=0 para este heatmap
    )
    for j, f in enumerate(freqs):
        T_map[i, j] = calculate_transmission(f, ALPHA, structure, basis="linear")[_T_INDEX[T_COMPONENT]]

print("  100% — listo")

FREQ_GRID, THICK_GRID = np.meshgrid(freqs, thicknesses)

# ---------------------------------------------------------------------------
# Estilo TFG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           15,
    "axes.labelsize":      18,
    "xtick.labelsize":     14,
    "ytick.labelsize":     14,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
})

if PLOT_TYPE == "3d":
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        FREQ_GRID, THICK_GRID, T_map,
        cmap="viridis", linewidth=0, antialiased=True, alpha=0.95,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(f"$T_{{{_label}}}$", fontsize=14)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", labelpad=10)
    ax.set_ylabel("Espesor (nm)", labelpad=10)
    ax.set_zlabel(f"$T_{{{_label}}}$", labelpad=8)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
else:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.pcolormesh(freqs, thicknesses, T_map, cmap="viridis", shading="auto",
                       rasterized=True)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"$T_{{{_label}}}$", fontsize=18)
    cbar.outline.set_linewidth(0.9)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel("Espesor (nm)")

fig.tight_layout()

out = ROOT_PATH / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fname = f"mapa_T{_label}_espesor_{MATERIAL.__name__}_{SUBSTRATE.__name__}"
fig.savefig(out / f"{fname}.png", dpi=300, bbox_inches="tight")
fig.savefig(out / f"{fname}.pdf", bbox_inches="tight")
print(f"Guardado en: {out / (fname + '.png')}")

plt.show()
