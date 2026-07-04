# -*- coding: utf-8 -*-
"""
Reflexion TMM — estructura arbitraria de N capas.
Basado en Plot.ipynb (T4X4Project).

Edita la seccion de parametros para definir tu estructura.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, Air_gap, Glass, BaF2, SiO2,
    Au, Ag,
    MoO3, V2O5, MgTeMoO6, hBN,
    LayeredStructure,
    calculate_reflection,
)

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------

# Angulo de incidencia
alpha = 0.0 * np.pi / 180.0   # Si es 0.0, es incidencia normal.

# Superstrato y sustrato
superstrate = Air()
substrate   = BaF2()

# Mesh de frecuencias
N      = 1000
omegas = [600, 1100]   # cm^-1
omega  = np.linspace(omegas[0], omegas[1], N)

# Capas intermedias (d en metros, phi en radianes)
layers = [
    MoO3(d=1000e-9, phi=0.0),
]

structure = LayeredStructure(
    superstrate=superstrate,
    substrate=substrate,
    layers=layers,
)

# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------

R = np.zeros((N, 4))   # Rpp, Rss, Rps, Rsp

for i in range(N):
    r = calculate_reflection(omega[i], alpha, structure)
    for j in range(4):
        R[i, j] = r[j]

# ---------------------------------------------------------------------------
# Plot — estilo TFG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           13,
    "axes.labelsize":      14,
    "xtick.labelsize":     12,
    "ytick.labelsize":     12,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     12,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

# Paleta fija (orden validado): xx azul, yy aqua, xy amarillo, yx verde
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]

fig, ax = plt.subplots(figsize=(9, 4.8))

ax.plot(omega, R[:, 0], color=COLORS[0], lw=1.8, label=r"$R_{xx}$")
ax.plot(omega, R[:, 1], color=COLORS[1], lw=1.8, label=r"$R_{yy}$")
ax.plot(omega, R[:, 2], color=COLORS[2], lw=1.6, linestyle="--", label=r"$R_{xy}$")
ax.plot(omega, R[:, 3], color=COLORS[3], lw=1.6, linestyle="--", label=r"$R_{yx}$")
ax.set_ylabel("Reflectancia")
ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax.set_xlim(omega[0], omega[-1])
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()
