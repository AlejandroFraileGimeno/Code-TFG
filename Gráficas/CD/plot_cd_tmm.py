# -*- coding: utf-8 -*-
"""
Plot CD y R_total con TMM para una estructura bicapa cualquiera.
Estructura: Air / capa1(d1, phi=theta) / capa2(d2) / Au

CONFIG — modifica aquí los parámetros de la estructura.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, Au,
    MoO3, V2O5, MgTeMoO6,
    LayeredStructure,
    calculate_circular_dichroism_ref,
)

# ---------------------------------------------------------------------------
# CONFIG — modifica aquí
# ---------------------------------------------------------------------------
CAPA1    = MoO3       # Material capa 1 (rotada): MoO3, V2O5, MgTeMoO6
CAPA2    = MoO3       # Material capa 2: MoO3, V2O5, MgTeMoO6
THETA    = 30.0       # Angulo de rotacion capa 1 (grados)
D1       = 800.0      # Espesor capa 1 (nm)
D2       = 800.0      # Espesor capa 2 (nm)

FREQ_MIN = 600        # cm-1
FREQ_MAX = 1100       # cm-1
N_FREQS  = 1000
# ---------------------------------------------------------------------------

freqs  = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
lambdas_um = 1e4 / freqs  # conversion a micras

structure = LayeredStructure(
    superstrate=Air(),
    substrate=Au(),
    layers=[
        CAPA1(d=D1 * 1e-9, phi=np.deg2rad(THETA)),
        CAPA2(d=D2 * 1e-9),
    ],
)

print(f"Calculando TMM ({N_FREQS} puntos)...")
results     = [calculate_circular_dichroism_ref(f, 0, structure) for f in freqs]
CD          = np.array([r[1]  for r in results])   # con signo
CD_abs      = np.abs(CD)
R_total     = np.array([r[2]  for r in results])

idx_max = int(np.argmax(CD_abs))
print(f"  |CD| max = {CD_abs[idx_max]:.4f}  @  {freqs[idx_max]:.1f} cm-1  ({lambdas_um[idx_max]:.2f} um)")
print(f"  R_total en ese punto = {R_total[idx_max]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

# Eje superior: numero de onda
ax1 = axes[0]
ax1.plot(freqs, CD_abs, color="tab:blue",   lw=1.5, label="|CD|")
ax1.plot(freqs, CD,     color="tab:blue",   lw=0.8, ls="--", alpha=0.5, label="CD (con signo)")
ax1.plot(freqs, R_total, color="tab:orange", lw=1.2, label="R_total")
ax1.set_xlabel("Numero de onda (cm-1)", fontsize=12)
ax1.set_ylabel("CD / R_total", fontsize=12)
ax1.set_xlim(FREQ_MIN, FREQ_MAX)
ax1.set_ylim(0, None)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axvline(freqs[idx_max], color="gray", ls=":", lw=1)

# Eje inferior: longitud de onda (micras) — eje invertido
ax2 = axes[1]
ax2.plot(lambdas_um, CD_abs,  color="tab:blue",   lw=1.5, label="|CD|")
ax2.plot(lambdas_um, R_total, color="tab:orange",  lw=1.2, label="R_total")
ax2.set_xlabel("Longitud de onda (um)", fontsize=12)
ax2.set_ylabel("CD / R_total", fontsize=12)
ax2.invert_xaxis()
ax2.set_ylim(0, None)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

titulo = (
    f"{CAPA1.__name__}({D1:.0f} nm, th={THETA:.0f} deg) / {CAPA2.__name__}({D2:.0f} nm) / Au\n"
    f"|CD|_max = {CD_abs[idx_max]:.4f}  @  {freqs[idx_max]:.1f} cm-1  ({lambdas_um[idx_max]:.2f} um)"
)
fig.suptitle(titulo, fontsize=11)
fig.tight_layout()
plt.show()
