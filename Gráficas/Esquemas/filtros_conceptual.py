# -*- coding: utf-8 -*-
"""
Esquema conceptual de los tres tipos de filtro trabajados en el TFG:
  (a) Notch      — rechazo estrecho en una única frecuencia (dip gaussiano)
  (b) Paso alto  — transición sigmoide, rechaza bajas frecuencias
  (c) Band-stop  — rechazo de una banda completa, paso fuera de ella

Las tres formas funcionales son idénticas a las usadas como target en
Tandem/T_xx/.../Prediccion_Gaussiana, Prediccion_FiltroEdge y
Prediccion_Stopband — esto es una ilustración conceptual, no una simulación.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           15,
    "axes.labelsize":      18,
    "axes.titlesize":      15,
    "xtick.labelsize":     14,
    "ytick.labelsize":     14,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

COLOR_T    = "#0b0b0b"   # curva (rol: verdad/objetivo)
COLOR_BAND = "#e1e0d9"   # banda de rechazo (rol: neutro)


def _sig(w, center, width):
    return 1.0 / (1.0 + np.exp(-(w - center) / (width / 6.0)))


def add_panel_label(ax, label):
    ax.text(0.04, 0.94, f"({label})", transform=ax.transAxes,
             ha="left", va="top", fontsize=15, fontweight="bold",
             bbox=dict(facecolor="0.88", edgecolor="none", pad=3))


fig, (ax_notch, ax_hp, ax_bs) = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)

# ---------------------------------------------------------------------------
# (a) Notch — dip gaussiano en omega_0
# ---------------------------------------------------------------------------
w0, sigma = 800.0, 15.0
w = np.linspace(600, 1000, 1000)
T = 1.0 - np.exp(-0.5 * ((w - w0) / sigma) ** 2)

ax_notch.axvspan(w0 - 3 * sigma, w0 + 3 * sigma, color=COLOR_BAND, zorder=0)
ax_notch.plot(w, T, color=COLOR_T, lw=2.2)
ax_notch.axvline(w0, color="0.4", ls=":", lw=1.2)
ax_notch.set_xlim(w[0], w[-1])
ax_notch.set_xticks([w0])
ax_notch.set_xticklabels([r"$\omega_0$"])
add_panel_label(ax_notch, "a")

# ---------------------------------------------------------------------------
# (b) Paso alto — sigmoide simple en omega_c
# ---------------------------------------------------------------------------
wc, flank = 800.0, 40.0
w = np.linspace(600, 1000, 1000)
T = _sig(w, wc, flank)

ax_hp.axvspan(w[0], wc, color=COLOR_BAND, zorder=0)
ax_hp.plot(w, T, color=COLOR_T, lw=2.2)
ax_hp.axvline(wc, color="0.4", ls=":", lw=1.2)
ax_hp.set_xlim(w[0], w[-1])
ax_hp.set_xticks([wc])
ax_hp.set_xticklabels([r"$\omega_c$"])
add_panel_label(ax_hp, "b")

# ---------------------------------------------------------------------------
# (c) Band-stop — producto de dos sigmoides (rechazo entre wL y wR)
# ---------------------------------------------------------------------------
wL, wR, flank_bs = 800.0, 950.0, 25.0
w = np.linspace(600, 1150, 1000)
inside = _sig(w, wL, flank_bs) * (1.0 - _sig(w, wR, flank_bs))
T = 1.0 - inside

ax_bs.axvspan(wL, wR, color=COLOR_BAND, zorder=0)
ax_bs.plot(w, T, color=COLOR_T, lw=2.2)
ax_bs.axvline(wL, color="0.4", ls=":", lw=1.2)
ax_bs.axvline(wR, color="0.4", ls=":", lw=1.2)
ax_bs.set_xlim(w[0], w[-1])
ax_bs.set_xticks([wL, wR])
ax_bs.set_xticklabels([r"$\omega_L$", r"$\omega_R$"])
add_panel_label(ax_bs, "c")

# ---------------------------------------------------------------------------
# Ajustes comunes
# ---------------------------------------------------------------------------
for ax in (ax_notch, ax_hp, ax_bs):
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(r"$\omega$")
    ax.grid(True, axis="y")

ax_notch.set_ylabel("Transmitancia")

fig.tight_layout()

out = Path(__file__).resolve().parents[2] / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fig.savefig(out / "filtros_notch_highpass_bandstop.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "filtros_notch_highpass_bandstop.pdf", bbox_inches="tight")
print(f"Guardado en: {out}")
plt.show()
