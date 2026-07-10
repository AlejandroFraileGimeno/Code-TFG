# -*- coding: utf-8 -*-
"""
Birrefringencia en el plano de MgTeMoO6 (potencial como retardador natural).

Delta n = n_x - n_y, Delta kappa = kappa_x - kappa_y, calculados a partir
del modelo dieléctrico propio (eps_XYZ_MgTeMoO6, ejes "X"/"Y") ya usado en
el resto del TFG.
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
COLOR_DN = "#2a78d6"
COLOR_DK = "#e34948"

def eps_x_MgTeMoO6(wl):
    return eps_XYZ_MgTeMoO6(wl, "X")

def eps_y_MgTeMoO6(wl):
    return eps_XYZ_MgTeMoO6(wl, "Y")

# ---------------------------------------------------------------------------
# Cálculo: n, kappa por eje a partir de eps_XYZ_MgTeMoO6("X") / ("Y")
# ---------------------------------------------------------------------------
OMEGA = np.linspace(600, 1100, 2000)   # cm-1

n_x = np.empty_like(OMEGA)
k_x = np.empty_like(OMEGA)
n_y = np.empty_like(OMEGA)
k_y = np.empty_like(OMEGA)

for i, w in enumerate(OMEGA):
    wl = convert_to_wavelength(w)
    nx_complex = np.sqrt(eps_x_MgTeMoO6(wl))
    ny_complex = np.sqrt(eps_y_MgTeMoO6(wl))
    n_x[i], k_x[i] = nx_complex.real, nx_complex.imag
    n_y[i], k_y[i] = ny_complex.real, ny_complex.imag

dn = n_x - n_y
dk = k_x - k_y

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

ax.axhline(0, color="0.5", lw=0.8, ls=":")
ax.plot(OMEGA, dn, color=COLOR_DN, lw=2.6, label=r"$\Delta n = n_x - n_y$")
ax.plot(OMEGA, dk, color=COLOR_DK, lw=2.6, label=r"$\Delta \kappa = \kappa_x - \kappa_y$")

ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax.set_ylabel("Birrefringencia")
ax.set_xlim(OMEGA[0], OMEGA[-1])
ax.legend(loc="best")
ax.grid(True, which="both")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax.set_title(
    r"MgTeMoO$_6$ - Birrefringencia",
    fontsize=18, fontweight="bold", pad=10,
)

fig.tight_layout()

out = ROOT / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fig.savefig(out / "birrefringencia_MgTeMoO6.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "birrefringencia_MgTeMoO6.pdf", bbox_inches="tight")
print(f"Guardado en: {out}")
plt.show()
