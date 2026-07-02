# -*- coding: utf-8 -*-
"""
Atlas de bandas Reststrahlen (ejes in-plane) para materiales vdW del Mid-IR.
Versión mejorada de la Fig. 5a del artículo: muestra la banda completa (TO→LO)
en lugar de un solo punto en TO.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method.permittivities import eps_XYZ_MgTeMoO6
from generalized_transfer_matrix_method.helpers import convert_to_wavelength

# ─────────────────────────────────────────────────────────────────────────────
#  Modelos de permitividad (parámetros de plot_eps_combinado.py)
# ─────────────────────────────────────────────────────────────────────────────

def eps_tolo_multi(w, eps_inf, modes):
    eps = eps_inf * np.ones_like(w, dtype=complex)
    for m in modes:
        eps *= (m["wL"]**2 - w**2 - 1j*m["g"]*w) / (
               m["wT"]**2 - w**2 - 1j*m["g"]*w)
    return eps

def eps_tolo(w, eps_inf, wT, wL, g):
    return eps_inf * (wL**2 - w**2 - 1j*g*w) / (wT**2 - w**2 - 1j*g*w)

def eps_hbn_perp(w):
    """hBN in-plane (eje x = eje y, uniaxial)."""
    eps_inf, wTO, wLO, gamma = 4.87, 1372.0, 1610.0, 5.0
    return eps_inf + eps_inf * (wLO**2 - wTO**2) / (wTO**2 - w**2 - 1j*gamma*w)

# ─────────────────────────────────────────────────────────────────────────────
#  Cálculo de permitividades
# ─────────────────────────────────────────────────────────────────────────────

OMEGA_MIN, OMEGA_MAX = 400, 1650
w    = np.linspace(OMEGA_MIN, OMEGA_MAX, 15000)
wl_m = np.array([convert_to_wavelength(wi) for wi in w])

eps = {
    r"$\alpha$-MoO$_3$": {
        "x": eps_tolo_multi(w, 5.78, [
                {"wT": 506.7, "wL": 534.3, "g": 49.1},
                {"wT": 821.4, "wL": 963.0, "g": 6.0},
                {"wT": 998.7, "wL": 999.2, "g": 0.35},
             ]),
        "y": eps_tolo_multi(w, 6.07, [
                {"wT": 544.6, "wL": 850.1, "g": 9.5},
             ]),
    },
    r"$\alpha$-V$_2$O$_5$": {
        "x": eps_tolo(w, 6.559, wT=770.0, wL=944.3, g=8.1),
        "y": eps_tolo(w, 6.142, wT=474.4, wL=815.6, g=9.6),
    },
    "hBN": {
        "x": eps_hbn_perp(w),
        "y": None,   # uniaxial: εx = εy
    },
    "MgTeMoO$_6$": {
        "x": np.array([eps_XYZ_MgTeMoO6(wl, "X") for wl in wl_m]),
        "y": np.array([eps_XYZ_MgTeMoO6(wl, "Y") for wl in wl_m]),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Detección de bandas Reststrahlen (Re(ε) < 0)
# ─────────────────────────────────────────────────────────────────────────────

def find_bands(w_arr, eps_r, gap=15):
    """Detecta regiones Re(ε) < 0; une huecos menores de `gap` cm⁻¹."""
    neg, bands, start = eps_r < 0, [], None
    for i, v in enumerate(neg):
        if v and start is None:
            start = w_arr[i]
        elif not v and start is not None:
            if bands and (w_arr[i-1] - bands[-1][1]) < gap:
                bands[-1] = (bands[-1][0], w_arr[i-1])
            else:
                bands.append((start, w_arr[i-1]))
            start = None
    if start is not None:
        bands.append((start, w_arr[-1]))
    return bands

RB = {}
for name, data in eps.items():
    RB[name] = {
        "x": find_bands(w, np.real(data["x"])),
        "y": find_bands(w, np.real(data["y"])) if data["y"] is not None else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Colores y estilos
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    r"$\alpha$-MoO$_3$":    "#2166ac",   # azul
    r"$\alpha$-V$_2$O$_5$": "#d6604d",   # rojo-naranja
    "hBN":                   "#1a9641",   # verde
    "MgTeMoO$_6$":           "#762a83",   # morado
}

# ─────────────────────────────────────────────────────────────────────────────
#  Figura
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "cm",
    "font.size":         12,
    "axes.linewidth":    0.9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "xtick.minor.visible": True,
})

fig, ax = plt.subplots(figsize=(9, 4.0))

materials = list(RB.keys())
n = len(materials)
y_centers = np.arange(n, 0, -1, dtype=float)  # 4, 3, 2, 1

BAR_H  = 0.22    # semi-altura de cada barra
OFFSET = 0.28    # separación vertical entre barra x e y

for idx, name in enumerate(materials):
    yc    = y_centers[idx]
    color = COLORS[name]
    has_y = RB[name]["y"] is not None

    y_x = yc + (OFFSET if has_y else 0)
    y_y = yc - OFFSET

    # ── bandas εx (sólido) ──────────────────────────────────────
    for (b0, b1) in RB[name]["x"]:
        ax.barh(y_x, b1 - b0, left=b0, height=2 * BAR_H,
                color=color, alpha=0.88, edgecolor="white", linewidth=0.6,
                zorder=2)
        ax.plot(b0, y_x, marker=7, color=color, markersize=6, zorder=3,
                clip_on=False)   # marker 7 = CARETRIGHT (triángulo ►)

    # ── bandas εy (tramado) ─────────────────────────────────────
    if has_y:
        for (b0, b1) in RB[name]["y"]:
            ax.barh(y_y, b1 - b0, left=b0, height=2 * BAR_H,
                    color=color, alpha=0.30, edgecolor=color, linewidth=0.9,
                    hatch="////", zorder=2)
            ax.plot(b0, y_y, marker=7, color=color, markersize=6,
                    alpha=0.55, zorder=3, clip_on=False)

# ─────────────────────────────────────────────────────────────────────────────
#  Etiquetas εx / εy a la derecha
# ─────────────────────────────────────────────────────────────────────────────

X_LABEL = OMEGA_MAX + 18
for idx, name in enumerate(materials):
    yc    = y_centers[idx]
    color = COLORS[name]
    has_y = RB[name]["y"] is not None
    if has_y:
        ax.text(X_LABEL, yc + OFFSET, r"$\varepsilon_x$",
                va="center", ha="left", fontsize=9.5, color=color)
        ax.text(X_LABEL, yc - OFFSET, r"$\varepsilon_y$",
                va="center", ha="left", fontsize=9.5, color=color, alpha=0.7)
    else:
        ax.text(X_LABEL, yc + OFFSET, r"$\varepsilon_{x{=}y}$",
                va="center", ha="left", fontsize=9.5, color=color)

# ─────────────────────────────────────────────────────────────────────────────
#  Línea de referencia en 820 cm⁻¹ (filtro notch del artículo)
# ─────────────────────────────────────────────────────────────────────────────

ax.axvline(820, color="dimgray", lw=0.9, ls="--", zorder=1, alpha=0.7)
ax.text(820, n + 0.52, r"$820\ \mathrm{cm}^{-1}$",
        ha="center", va="bottom", fontsize=8.5, color="dimgray", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
#  Ejes y formato
# ─────────────────────────────────────────────────────────────────────────────

ax.set_yticks(y_centers)
ax.set_yticklabels(materials, fontsize=12)
ax.set_ylim(0.35, n + 0.65)

ax.set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$", fontsize=13)
ax.set_xlim(OMEGA_MIN, OMEGA_MAX)
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.tick_params(which="major", length=5, width=0.9)
ax.tick_params(which="minor", length=2.5, width=0.7)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_tick_params(length=0)

ax.grid(axis="x", alpha=0.15, zorder=0, linestyle="--")

# ─────────────────────────────────────────────────────────────────────────────
#  Leyenda
# ─────────────────────────────────────────────────────────────────────────────

legend_elems = [
    mpatches.Patch(facecolor="gray", alpha=0.88, edgecolor="white",
                   label=r"$\varepsilon_x$ (banda Reststrahlen)"),
    mpatches.Patch(facecolor="gray", alpha=0.30, edgecolor="gray",
                   hatch="////", label=r"$\varepsilon_y$ (banda Reststrahlen)"),
]
ax.legend(handles=legend_elems, loc="upper left", frameon=True,
          framealpha=0.92, fontsize=9.5, edgecolor="0.75")

fig.tight_layout()
plt.show()
