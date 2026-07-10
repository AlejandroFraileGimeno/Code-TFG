# -*- coding: utf-8 -*-
"""
Espectro de transmitancia T_xx de una lámina de MoO3 (d = 2000 nm) sobre BaF2.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, LayeredStructure, calculate_transmission,
)

plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           15,
    "axes.labelsize":      16,
    "axes.titlesize":      15,
    "xtick.labelsize":     13,
    "ytick.labelsize":     13,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     13,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

COLOR_T = "#0b0b0b"

D_NM = 2000.0

# ---------------------------------------------------------------------------
# Cálculo TMM
# ---------------------------------------------------------------------------
FREQS = np.linspace(400, 1400, 1500)   # cm-1

structure = LayeredStructure(
    superstrate=Air(), substrate=BaF2(),
    layers=[MoO3(d=D_NM * 1e-9)],
)

T = np.array([
    float(calculate_transmission(f, 0, structure, basis="linear")[0]) for f in FREQS
])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))

ax.plot(FREQS, T, color=COLOR_T, lw=2.0)

ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax.set_ylabel(r"$T_{xx}$")
ax.set_xlim(FREQS[0], FREQS[-1])
ax.set_ylim(-0.02, 1.05)
ax.grid(True, which="both")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax.set_title(
    r"MoO$_3$ - Transmitancia",
    fontsize=15, fontweight="bold", pad=10,
)

fig.tight_layout()

out = ROOT / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fig.savefig(out / "transmitancia_MoO3.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "transmitancia_MoO3.pdf", bbox_inches="tight")
print(f"Guardado en: {out}")
plt.show()
