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
T_COMPONENT = "pp"        # Componente: "pp", "ss", "ps", "sp"
PLOT_TYPE   = "heatmap"   # "heatmap" o "3d"
AZIMUTH     = -60         # Ángulo azimutal para la vista 3D (grados)
ELEVATION   = 30          # Ángulo de elevación para la vista 3D (grados)
# ---------------------------------------------------------------------------

if len(sys.argv) > 1:  # permite: python Waterplot_phi.py ss
    T_COMPONENT = sys.argv[1]

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
        FREQ_GRID, PHI_GRID, T_map,
        cmap="viridis", linewidth=0, antialiased=True, alpha=0.95,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(f"$T_{{{_label}}}$", fontsize=14)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", labelpad=10)
    ax.set_ylabel(r"$\phi$ (°)", labelpad=10)
    ax.set_zlabel(f"$T_{{{_label}}}$", labelpad=8)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
else:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.pcolormesh(freqs, phis, T_map, cmap="viridis", shading="auto",
                       rasterized=True)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"$T_{{{_label}}}$", fontsize=18)
    cbar.outline.set_linewidth(0.9)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel(r"$\phi$ (°)")
    ax.set_yticks(np.arange(PHI_MIN, PHI_MAX + 1, 15))

fig.tight_layout()

out = ROOT_PATH / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fname = f"mapa_T{_label}_phi_{MATERIAL.__name__}_{SUBSTRATE.__name__}"
fig.savefig(out / f"{fname}.png", dpi=300, bbox_inches="tight")
fig.savefig(out / f"{fname}.pdf", bbox_inches="tight")
print(f"Guardado en: {out / (fname + '.png')}")

plt.show()
