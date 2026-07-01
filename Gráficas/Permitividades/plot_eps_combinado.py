# -*- coding: utf-8 -*-
"""
Figura combinada en cascada (6 filas, eje x compartido 400-1700 cm^-1):
  (a) α-MoO3   Re(εx)        (b) α-MoO3   Re(εy)
  (c) α-V2O5   Re(εx)        (d) α-V2O5   Re(εy)
  (e) MgTeMoO6 Re(εx)        (f) MgTeMoO6 Re(εy)
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
#  Modelos
# ============================================================

def epsilon_tolo(w, eps_inf, w_TO, w_LO, gamma):
    return eps_inf * (w_LO**2 - w**2 - 1j*gamma*w) / (
                     w_TO**2 - w**2 - 1j*gamma*w)

def epsilon_tolo_multi(w, eps_inf, modes):
    eps = eps_inf * np.ones_like(w, dtype=complex)
    for m in modes:
        eps *= (m["w_LO"]**2 - w**2 - 1j*m["gamma"]*w) / (
               m["w_TO"]**2 - w**2 - 1j*m["gamma"]*w)
    return eps

# ============================================================
#  Parámetros
# ============================================================

MoO3 = {
    "x": {"eps_inf": 5.78, "modes": [
        {"w_TO": 506.7,  "w_LO": 534.3,  "gamma": 49.1},
        {"w_TO": 821.4,  "w_LO": 963.0,  "gamma": 6.0},
        {"w_TO": 998.7,  "w_LO": 999.2,  "gamma": 0.35},
    ]},
    "y": {"eps_inf": 6.07, "modes": [{"w_TO": 544.6, "w_LO": 850.1, "gamma": 9.5}]},
}

V2O5 = {
    "x": {"eps_inf": 6.559, "w_TO": 770.0,  "w_LO": 944.3, "gamma": 8.1},
    "y": {"eps_inf": 6.142, "w_TO": 474.4,  "w_LO": 815.6, "gamma": 9.6},
}

# ============================================================
#  Frecuencias comunes
# ============================================================

XMIN, XMAX = 400, 1700
w    = np.linspace(XMIN, XMAX, 10000)
wl_m = np.array([convert_to_wavelength(wi) for wi in w])

ex_a = epsilon_tolo_multi(w, **MoO3["x"])
ey_a = epsilon_tolo_multi(w, **MoO3["y"])

ex_b = epsilon_tolo(w, **V2O5["x"])
ey_b = epsilon_tolo(w, **V2O5["y"])

ex_c = eps_XYZ_MgTeMoO6(wl_m, "X")
ey_c = eps_XYZ_MgTeMoO6(wl_m, "Y")

# ============================================================
#  Detección de bandas Reststrahlen (Re(ε) < 0)
# ============================================================

def bandas_neg(w, er):
    neg, out, start = er < 0, [], None
    for i, v in enumerate(neg):
        if v and start is None:
            start = w[i]
        elif not v and start is not None:
            out.append((start, w[i-1])); start = None
    if start is not None:
        out.append((start, w[-1]))
    return out

def cluster(bands, gap=15):
    out = []
    for b0, b1 in bands:
        if out and b0 - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], b1))
        else:
            out.append((b0, b1))
    return out

# MoO3: usar TO-LO explícitos para etiquetas físicas
BANDS_moo3_x = [(821.4, 963.0)]       # RB₂ (banda principal; la de 506-534 es estrecha)
BANDS_moo3_y = [(544.6, 850.1)]       # RB₁
LBL_moo3_x   = [r"RB$_2$"]
LBL_moo3_y   = [r"RB$_1$"]

# Banda estrecha de MoO3 εx (506-534) — sombreada pero sin etiqueta
EXTRA_moo3_x = bandas_neg(w, np.real(ex_a))   # detecta la estrecha también
EXTRA_moo3_x = [b for b in EXTRA_moo3_x if b[1] < 600]   # sólo la < 600 cm⁻¹

# V2O5: TO-LO explícitos
BANDS_v2o5_x = [(770.0, 944.3)]       # RB₂
BANDS_v2o5_y = [(474.4, 815.6)]       # RB₃
LBL_v2o5_x   = [r"RB$_2$"]
LBL_v2o5_y   = [r"RB$_3$"]

# MgTeMoO6: detección numérica + clustering por eje
BANDS_mg_x = cluster(bandas_neg(w, np.real(ex_c)))
BANDS_mg_y = cluster(bandas_neg(w, np.real(ey_c)))

# ============================================================
#  Estilos de sombreado (distintos por banda dentro del panel)
# ============================================================

S = [
    {"facecolor": "0.90", "edgecolor": "0.55", "hatch": "///"},   # 1ª banda
    {"facecolor": "0.77", "edgecolor": "0.45", "hatch": "\\\\"},  # 2ª banda
    {"facecolor": "0.64", "edgecolor": "0.35", "hatch": "||"},    # 3ª banda
]
S_EXTRA = {"facecolor": "0.88", "edgecolor": "0.55", "hatch": "..."}  # banda extra sin etiqueta

def shade(ax, band, style):
    ax.axvspan(*band, linewidth=0.0, zorder=0, **style)

# ============================================================
#  Estilo global
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

LW   = 1.5
YLIM = (-300, 300)

# ============================================================
#  Figura: 6 filas, eje x compartido
# ============================================================

fig, axes = plt.subplots(6, 1, figsize=(9, 16), sharex=True,
                         gridspec_kw={"hspace": 0.07})

def base(ax, label, curve, clabel):
    ax.plot(w, np.real(curve), "k-", lw=LW, label=clabel, zorder=2)
    ax.axhline(0, color="black", lw=0.8, zorder=1.5)
    ax.set_ylim(*YLIM)
    ax.set_ylabel(r"$\mathrm{Re}(\varepsilon)$")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.text(0.01, 0.96, label, transform=ax.transAxes, fontsize=11, va="top")
    ax.legend(frameon=False, loc="center right", handlelength=2.0, fontsize=9.5)

def rb_labels(ax, bands, labels, y=248):
    for (b0, b1), lbl in zip(bands, labels):
        ax.text(0.5*(b0+b1), y, lbl, ha="center", va="top",
                fontsize=10, fontweight="bold")

# ── (a) α-MoO3 — εx ─────────────────────────────────────────

ax = axes[0]
for b in EXTRA_moo3_x:
    shade(ax, b, S_EXTRA)           # banda estrecha 506-534 sin etiqueta
for i, b in enumerate(BANDS_moo3_x):
    shade(ax, b, S[i])
base(ax, r"(a) $\alpha$-MoO$_3$ — $\varepsilon_x$",
     ex_a, r"$\mathrm{Re}(\varepsilon_x)$")
rb_labels(ax, BANDS_moo3_x, LBL_moo3_x)

# ── (b) α-MoO3 — εy ─────────────────────────────────────────

ax = axes[1]
for i, b in enumerate(BANDS_moo3_y):
    shade(ax, b, S[i])
base(ax, r"(b) $\alpha$-MoO$_3$ — $\varepsilon_y$",
     ey_a, r"$\mathrm{Re}(\varepsilon_y)$")
rb_labels(ax, BANDS_moo3_y, LBL_moo3_y)

# ── (c) α-V2O5 — εx ─────────────────────────────────────────

ax = axes[2]
for i, b in enumerate(BANDS_v2o5_x):
    shade(ax, b, S[i])
base(ax, r"(c) $\alpha$-V$_2$O$_5$ — $\varepsilon_x$",
     ex_b, r"$\mathrm{Re}(\varepsilon_x)$")
rb_labels(ax, BANDS_v2o5_x, LBL_v2o5_x)

# ── (d) α-V2O5 — εy ─────────────────────────────────────────

ax = axes[3]
for i, b in enumerate(BANDS_v2o5_y):
    shade(ax, b, S[i])
base(ax, r"(d) $\alpha$-V$_2$O$_5$ — $\varepsilon_y$",
     ey_b, r"$\mathrm{Re}(\varepsilon_y)$")
rb_labels(ax, BANDS_v2o5_y, LBL_v2o5_y)

# ── (e) MgTeMoO6 — εx ────────────────────────────────────────

ax = axes[4]
for i, b in enumerate(BANDS_mg_x):
    shade(ax, b, S[i % len(S)])
base(ax, r"(e) MgTeMoO$_6$ — $\varepsilon_x$",
     ex_c, r"$\mathrm{Re}(\varepsilon_x)$")
rb_labels(ax, BANDS_mg_x, [rf"RB$_{n}$" for n in range(1, len(BANDS_mg_x)+1)])

# ── (f) MgTeMoO6 — εy ────────────────────────────────────────

ax = axes[5]
for i, b in enumerate(BANDS_mg_y):
    shade(ax, b, S[i % len(S)])
base(ax, r"(f) MgTeMoO$_6$ — $\varepsilon_y$",
     ey_c, r"$\mathrm{Re}(\varepsilon_y)$")
rb_labels(ax, BANDS_mg_y, [rf"RB$_{n}$" for n in range(1, len(BANDS_mg_y)+1)])

# ── eje x común ──────────────────────────────────────────────

axes[-1].set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$")
axes[-1].set_xlim(XMIN, XMAX)
axes[-1].set_xticks(range(400, 1750, 100))

# ============================================================
#  Guardar
# ============================================================

out = Path(__file__).parent
fig.savefig(out / "eps_combinado.pdf", bbox_inches="tight")
fig.savefig(out / "eps_combinado.png", dpi=300, bbox_inches="tight")
print(f"Guardado en: {out}")
plt.show()