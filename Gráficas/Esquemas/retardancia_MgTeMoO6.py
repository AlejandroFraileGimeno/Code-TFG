# -*- coding: utf-8 -*-
"""
Retardancia (diferencia de fase) de MgTeMoO6 como retardador natural, d=1000 nm.

psi = 360 * d * Delta n * omega  (grados), con omega en cm-1 y d en cm.
Delta n = n_x - n_y, calculado a partir del modelo dieléctrico propio
(eps_XYZ_MgTeMoO6, ejes "X"/"Y") ya usado en el resto del TFG.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TMM"))

from generalized_transfer_matrix_method.permittivities import eps_XYZ_MgTeMoO6
from generalized_transfer_matrix_method.helpers import convert_to_wavelength

plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           17,
    "axes.labelsize":      19,
    "axes.titlesize":      18,
    "xtick.labelsize":     16,
    "ytick.labelsize":     16,
    "axes.linewidth":      1.1,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     15,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "grid.linewidth":      0.6,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

FIGSIZE = (5.8, 4.1)
COLOR_G   = "#2a78d6"
COLOR_REF = "#898781"

def eps_x_MgTeMoO6(wl):
    return eps_XYZ_MgTeMoO6(wl, "X")

def eps_y_MgTeMoO6(wl):
    return eps_XYZ_MgTeMoO6(wl, "Y")

D_NM = 1000.0
D_CM = D_NM * 1e-7   # nm -> cm

# ---------------------------------------------------------------------------
# Cálculo: n por eje a partir de eps_XYZ_MgTeMoO6("X") / ("Y")
# ---------------------------------------------------------------------------
OMEGA = np.linspace(600, 1100, 2000)   # cm-1

n_x = np.empty_like(OMEGA)
n_y = np.empty_like(OMEGA)

for i, w in enumerate(OMEGA):
    wl = convert_to_wavelength(w)
    n_x[i] = np.sqrt(eps_x_MgTeMoO6(wl)).real
    n_y[i] = np.sqrt(eps_y_MgTeMoO6(wl)).real

dn    = n_x - n_y
gamma = 360.0 * D_CM * dn * OMEGA   # grados

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

for val, lab in ((90, r"$\lambda/4$"), (180, r"$\lambda/2$"), (-90, r"$\lambda/4$"), (-180, r"$\lambda/2$")):
    ax.axhline(val, color=COLOR_REF, lw=1.1, ls="--")
    ax.text(OMEGA[-1], val, f"  {lab}", va="center", ha="left", fontsize=16, color=COLOR_REF)

ax.plot(OMEGA, gamma, color=COLOR_G, lw=2.6)

ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax.set_ylabel(r"Desfase $\psi$ ($^\circ$)")
ax.set_xlim(OMEGA[0], OMEGA[-1])
ax.grid(True, which="both")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax.set_title(
    r"MgTeMoO$_6$ - Desfase",
    fontsize=18, fontweight="bold", pad=10,
)

fig.tight_layout()

out = ROOT / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fig.savefig(out / "retardancia_MgTeMoO6.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "retardancia_MgTeMoO6.pdf", bbox_inches="tight")
print(f"Guardado en: {out}")
plt.show()
