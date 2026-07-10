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
    "font.size": 18,
    "axes.labelsize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ============================================================
#  Figura
# ============================================================

fig, ax = plt.subplots(figsize=(5.8, 4.2))

# Las resonancias de MgTeMoO6 llegan a |Re(eps)| ~ 135 (no a 300
# como MoO3/V2O5): se ajusta el rango para aprovechar la figura
ylim = (-160, 160)

for _, (b0, b1) in todas:
    ax.axvspan(b0, b1, facecolor="0.92", edgecolor="0.65",
               hatch="///", linewidth=0.0, zorder=0)

for b0, b1 in overlaps:
    ax.axvspan(b0, b1, facecolor="0.70", edgecolor="0.35",
               hatch="xxx", linewidth=0.0, zorder=0.2)

ax.plot(w, np.real(eps_x), color="black", linewidth=2.2, linestyle="-",
        label=r"$\mathrm{Re}(\varepsilon_x)$", zorder=2)
ax.plot(w, np.real(eps_y), color="black", linewidth=2.2, linestyle="--",
        label=r"$\mathrm{Re}(\varepsilon_y)$", zorder=2)
ax.plot(w, np.real(eps_z), color="black", linewidth=2.2, linestyle=":",
        label=r"$\mathrm{Re}(\varepsilon_z)$", zorder=2)

# ============================================================
#  Etiquetas RB (una por grupo de bandas solapadas, en orden
#  ascendente de frecuencia, como en MoO3)
# ============================================================

def agrupar(bandas, gap=15):
    bandas = sorted(bandas)
    grupos = []
    for b0, b1 in bandas:
        if grupos and b0 - grupos[-1][1] < gap:
            grupos[-1] = (grupos[-1][0], max(grupos[-1][1], b1))
        else:
            grupos.append((b0, b1))
    return grupos

grupos = agrupar([b for _, b in todas])
for n, (b0, b1) in enumerate(grupos, start=1):
    ax.text(
        0.5 * (b0 + b1),
        0.82 * ylim[1],
        rf"RB$_{n}$",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
    )

ax.axhline(0, color="black", linewidth=1.0, zorder=1.5)

ax.set_xlim(400, 1100)
ax.set_ylim(*ylim)
ax.set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$")
ax.set_ylabel(r"$\mathrm{Re}(\varepsilon)$")
# Ticks espaciados para que sigan siendo legibles al ampliar la fuente.
ax.set_xticks([400, 600, 800, 1000])
ax.minorticks_on()
ax.tick_params(which="major", direction="in", top=True, right=True, width=1.2, length=6)
ax.tick_params(which="minor", direction="in", top=True, right=True, width=1.0, length=3)

ax.legend(
    frameon=True,
    facecolor="white",
    edgecolor="0.55",
    framealpha=1.0,
    loc="lower right",
    handlelength=2.4,
)

# ============================================================
#  Guardar figura
# ============================================================

out_dir = Path(__file__).resolve().parents[2] / "Arreglos en Gráficos"
out_dir.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out_dir / "MgTeMoO6_Re_epsilon_Reststrahlen_BW.pdf", bbox_inches="tight")
fig.savefig(out_dir / "MgTeMoO6_Re_epsilon_Reststrahlen_BW.png", dpi=400, bbox_inches="tight")
print(f"Guardado en: {out_dir}")

plt.show()
