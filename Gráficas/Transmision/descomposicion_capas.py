# -*- coding: utf-8 -*-
"""
Descomposición de la bicapa V2O5/MoO3/BaF2 en sus dos capas por separado.
Estructura del diseño inverso (filtro gaussiano):
    Aire / V2O5(d1, th1) / MoO3(d2, th2) / BaF2

Se comparan tres respuestas T_xx (TMM):
    1. Bicapa completa   V2O5 / MoO3
    2. Solo V2O5         Aire / V2O5(d1, th1) / BaF2
    3. Solo MoO3         Aire / MoO3(d2, th2) / BaF2
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2,
    MoO3, V2O5,
    LayeredStructure,
    calculate_transmission,
)

# ---------------------------------------------------------------------------
# CONFIG — parámetros del diseño inverso
# ---------------------------------------------------------------------------
TH1 = 180.0    # V2O5 (capa 1, arriba)   — grados
D1  = 200.0    # V2O5                     — nm
TH2 = 177.0    # MoO3 (capa 2, sustrato) — grados
D2  = 261.0    # MoO3                     — nm

ALPHA    = 0.0     # incidencia normal (rad)
FREQ_MIN = 400     # cm^-1
FREQ_MAX = 1400    # cm^-1
N_FREQS  = 1200
# ---------------------------------------------------------------------------

omega = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)


def txx_spectrum(layers):
    """T_xx (índice 0, base lineal) de una estructura Aire / layers / BaF2."""
    structure = LayeredStructure(superstrate=Air(), substrate=BaF2(), layers=layers)
    return np.array([
        calculate_transmission(f, ALPHA, structure, basis="linear")[0]
        for f in omega
    ])


print("Calculando TMM (3 estructuras)...")
T_bicapa = txx_spectrum([V2O5(d=D1 * 1e-9, phi=np.deg2rad(TH1)),
                         MoO3(d=D2 * 1e-9, phi=np.deg2rad(TH2))])
T_v2o5   = txx_spectrum([V2O5(d=D1 * 1e-9, phi=np.deg2rad(TH1))])
T_moo3   = txx_spectrum([MoO3(d=D2 * 1e-9, phi=np.deg2rad(TH2))])
print("  listo")

# ---------------------------------------------------------------------------
# Estilo TFG (tipografía grande para el PDF)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           18,
    "axes.labelsize":      21,
    "axes.titlesize":      20,
    "xtick.labelsize":     17,
    "ytick.labelsize":     17,
    "axes.linewidth":      1.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":    6,
    "ytick.major.size":    6,
    "xtick.minor.size":    3,
    "ytick.minor.size":    3,
    "legend.fontsize":     17,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

COLOR_BICAPA = "#0b0b0b"   # negro — resultado (bicapa)
COLOR_V2O5   = "#2a78d6"   # azul
COLOR_MOO3   = "#e34948"   # rojo

fig, ax = plt.subplots(figsize=(7.6, 5.6))

ax.plot(omega, T_v2o5,   color=COLOR_V2O5, lw=2.0, ls="--",
        label=r"V$_2$O$_5$")
ax.plot(omega, T_moo3,   color=COLOR_MOO3, lw=2.0, ls="-.",
        label=r"MoO$_3$")
ax.plot(omega, T_bicapa, color=COLOR_BICAPA, lw=2.6, ls="-",
        label=r"V$_2$O$_5$ / MoO$_3$", zorder=3)

ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax.set_ylabel(r"$T_{xx}$")
ax.set_title(
    rf"$\alpha\mathrm{{-V_2O_5}}/\alpha\mathrm{{-MoO_3}}\ "
    rf"(d_1 = {D1:.0f}\,\mathrm{{nm}},\ d_2 = {D2:.0f}\,\mathrm{{nm}})$",
    pad=8,
)
ax.set_xlim(FREQ_MIN, FREQ_MAX)
ax.set_ylim(-0.02, 1.05)
ax.grid(True, which="both")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.legend(loc="lower right")

fig.tight_layout()

out = Path(__file__).parent / "descomposicion_capas.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
arreglo_out = ROOT_PATH / "Arreglos en Gráficos"
arreglo_out.mkdir(exist_ok=True)
fig.savefig(arreglo_out / "descomposicion_capas.png", dpi=300, bbox_inches="tight")
fig.savefig(arreglo_out / "descomposicion_capas.pdf", bbox_inches="tight")
print(f"Guardado: {out}")
plt.show()
