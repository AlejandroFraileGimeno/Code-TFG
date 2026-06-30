# -*- coding: utf-8 -*-
"""
Transmision TMM — estructura arbitraria de N capas.
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
    calculate_transmission,
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

T = np.zeros((N, 4))   # Tpp, Tss, Tps, Tsp

for i in range(N):
    t = calculate_transmission(omega[i], alpha, structure)
    for j in range(4):
        T[i, j] = t[j]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(omega, T[:, 0], label='Txx')
ax.plot(omega, T[:, 1], label='Tyy')
ax.plot(omega, T[:, 2], label='Tyx', linestyle='--')
ax.plot(omega, T[:, 3], label='Txy', linestyle='--')
ax.set_ylabel('Transmision', fontsize=13)
ax.set_xlabel('Frecuencia (cm$^{-1}$)', fontsize=13)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
