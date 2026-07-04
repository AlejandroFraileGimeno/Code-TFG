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
THETA    = 45      # Angulo de rotacion capa 1 (grados)
D1       = 500     # Espesor capa 1 (nm)
D2       = 500      # Espesor capa 2 (nm)

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
# Plot — estilo TFG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           12,
    "axes.labelsize":      13,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     11,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

COLOR_CD = "#2a78d6"
COLOR_R  = "#e34948"

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

# Eje superior: numero de onda
ax1 = axes[0]
ax1.plot(freqs, CD_abs,  color=COLOR_CD, lw=1.8, label=r"$|\mathrm{CD}|$")
ax1.plot(freqs, CD,      color=COLOR_CD, lw=0.9, ls="--", alpha=0.5,
         label="CD (con signo)")
ax1.plot(freqs, R_total, color=COLOR_R,  lw=1.4, label=r"$R_\mathrm{total}$")
ax1.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax1.set_ylabel(r"$|\mathrm{CD}|$ / $R_\mathrm{total}$")
ax1.set_xlim(FREQ_MIN, FREQ_MAX)
ax1.set_ylim(0, None)
ax1.legend()
ax1.axvline(freqs[idx_max], color="#898781", ls=":", lw=1)

# Eje inferior: longitud de onda (micras) — eje invertido
ax2 = axes[1]
ax2.plot(lambdas_um, CD_abs,  color=COLOR_CD, lw=1.8, label=r"$|\mathrm{CD}|$")
ax2.plot(lambdas_um, R_total, color=COLOR_R,  lw=1.4, label=r"$R_\mathrm{total}$")
ax2.set_xlabel(r"$\lambda$ ($\mu$m)")
ax2.set_ylabel(r"$|\mathrm{CD}|$ / $R_\mathrm{total}$")
ax2.invert_xaxis()
ax2.set_ylim(0, None)
ax2.legend()

titulo = (
    rf"{CAPA1.__name__}({D1:.0f} nm, $\theta$={THETA:.0f}°) / "
    rf"{CAPA2.__name__}({D2:.0f} nm) / Au"
    "\n"
    rf"$|\mathrm{{CD}}|_\mathrm{{max}}$ = {CD_abs[idx_max]:.4f}  @  "
    rf"{freqs[idx_max]:.1f} cm$^{{-1}}$  ({lambdas_um[idx_max]:.2f} $\mu$m)"
)
fig.suptitle(titulo, fontsize=12)
fig.tight_layout()
plt.show()
