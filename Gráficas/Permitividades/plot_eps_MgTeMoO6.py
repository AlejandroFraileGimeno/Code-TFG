# -*- coding: utf-8 -*-
"""
Re(eps_x), Re(eps_y), Re(eps_z) para MgTeMoO6 con bandas Reststrahlen.
Usa la permitividad real del TMM (no se reimplementa el modelo).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method.permittivities import eps_XYZ_MgTeMoO6
from generalized_transfer_matrix_method.helpers import convert_to_wavelength

# ============================================================
#  Ventana espectral
# ============================================================

w    = np.linspace(400, 1100, 7000)
wl_m = np.array([convert_to_wavelength(wi) for wi in w])

eps_x = eps_XYZ_MgTeMoO6(wl_m, "X")
eps_y = eps_XYZ_MgTeMoO6(wl_m, "Y")
eps_z = eps_XYZ_MgTeMoO6(wl_m, "Z")


# ============================================================
#  Deteccion de bandas Reststrahlen (Re(eps) < 0) por eje
# ============================================================

def bandas_negativas(w, eps_real):
    neg = eps_real < 0
    bandas = []
    start = None
    for i, v in enumerate(neg):
        if v and start is None:
            start = w[i]
        elif not v and start is not None:
            bandas.append((start, w[i - 1]))
            start = None
    if start is not None:
        bandas.append((start, w[-1]))
    return bandas


bandas_x = bandas_negativas(w, np.real(eps_x))
bandas_y = bandas_negativas(w, np.real(eps_y))
bandas_z = bandas_negativas(w, np.real(eps_z))

todas = (
    [("x", b) for b in bandas_x]
    + [("y", b) for b in bandas_y]
    + [("z", b) for b in bandas_z]
)
todas.sort(key=lambda t: t[1][0])

overlaps = []
for i in range(len(todas)):
    eje_i, (a0, a1) = todas[i]
    for j in range(i + 1, len(todas)):
        eje_j, (b0, b1) = todas[j]
        if eje_i == eje_j:
            continue
        lo, hi = max(a0, b0), min(a1, b1)
        if lo < hi:
            overlaps.append((lo, hi))


# ============================================================
#  Estilo general
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ============================================================
#  Figura
# ============================================================

fig, ax = plt.subplots(figsize=(7.6, 3.8))

ylim = (-300, 300)

for _, (b0, b1) in todas:
    ax.axvspan(b0, b1, facecolor="0.92", edgecolor="0.65",
               hatch="///", linewidth=0.0, zorder=0)

for b0, b1 in overlaps:
    ax.axvspan(b0, b1, facecolor="0.70", edgecolor="0.35",
               hatch="xxx", linewidth=0.0, zorder=0.2)

ax.plot(w, np.real(eps_x), color="black", linewidth=1.7, linestyle="-",
        label=r"$\mathrm{Re}(\varepsilon_x)$", zorder=2)
ax.plot(w, np.real(eps_y), color="black", linewidth=1.7, linestyle="--",
        label=r"$\mathrm{Re}(\varepsilon_y)$", zorder=2)
ax.plot(w, np.real(eps_z), color="black", linewidth=1.7, linestyle=":",
        label=r"$\mathrm{Re}(\varepsilon_z)$", zorder=2)

ax.axhline(0, color="black", linewidth=0.8, zorder=1.5)

ax.set_xlim(400, 1100)
ax.set_ylim(*ylim)
ax.set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$")
ax.set_ylabel(r"$\mathrm{Re}(\varepsilon)$")
ax.set_xticks([400, 500, 600, 700, 800, 900, 1000, 1100])
ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)

ax.legend(frameon=False, loc="lower right", handlelength=2.6)

# ============================================================
#  Guardar figura
# ============================================================

out_dir = Path(__file__).parent
fig.tight_layout()
fig.savefig(out_dir / "MgTeMoO6_Re_epsilon_Reststrahlen_BW.pdf", bbox_inches="tight")
fig.savefig(out_dir / "MgTeMoO6_Re_epsilon_Reststrahlen_BW.png", dpi=400, bbox_inches="tight")
print(f"Guardado en: {out_dir}")

plt.show()
