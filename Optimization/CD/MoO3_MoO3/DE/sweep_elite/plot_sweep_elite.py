# -*- coding: utf-8 -*-
"""
Visualización del barrido DE elite — MoO3/MoO3  (625–875 cm-1)

Carga todos los target_XXX.npz de results/ y genera:
  1. Heatmap:  eje X = número de onda, eje Y = λ_target, color = |CD|_TMM
  2. Parámetros óptimos (θ, d1, d2, FoM) vs λ_target
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Estilo TFG (solo estética; no afecta a los cálculos)
# ---------------------------------------------------------------------------
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
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     14,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Cargar resultados
# ---------------------------------------------------------------------------
files = sorted(
    RESULTS_DIR.glob("target_*.npz"),
    key=lambda f: float(f.stem.split("_")[1]),
)

if not files:
    print(f"No hay resultados en {RESULTS_DIR}")
    print("Ejecuta sweep_elite_paralelo.py primero.")
    sys.exit(1)

print(f"Cargando {len(files)} resultados...")
data        = [np.load(f) for f in files]
targets_cm  = np.array([float(d["target_cm"]) for d in data])
freqs       = data[0]["freqs"]
CD_matrix   = np.array([d["CD_tmm"]      for d in data])
R_matrix    = np.array([d["R_total_tmm"] for d in data])
thetas      = np.array([float(d["theta"]) for d in data])
d1s         = np.array([float(d["d1"])    for d in data])
d2s         = np.array([float(d["d2"])    for d in data])
foms        = np.array([float(d["fom_nn"]) for d in data])

print(f"  {len(targets_cm)} targets  |  "
      f"frecuencias {freqs[0]:.0f}–{freqs[-1]:.0f} cm⁻¹\n")

# ---------------------------------------------------------------------------
# Figura 1 — Heatmap |CD|_TMM
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(11, 6))

vmax = np.max(CD_matrix)
im   = ax1.pcolormesh(
    freqs, targets_cm, CD_matrix,
    cmap="inferno", shading="auto",
    vmin=0, vmax=vmax, rasterized=True,
)
cbar = fig1.colorbar(im, ax=ax1)
cbar.set_label(r"$|CD_{\mathrm{norm}}|$", fontsize=18)

f_diag = [max(freqs[0], targets_cm[0]), min(freqs[-1], targets_cm[-1])]
ax1.plot(f_diag, f_diag, "w--", lw=1.2, alpha=0.6, label=r"$\omega_{\mathrm{obj}} = \omega$")

ax1.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax1.set_ylabel(r"$\omega_{\mathrm{obj}}$ (cm$^{-1}$)")
ax1.legend(fontsize=14)
fig1.tight_layout()

out1 = RESULTS_DIR / "heatmap_CD_elite.png"
fig1.savefig(out1, dpi=200, bbox_inches="tight")
ARREGLOS = Path(__file__).resolve().parents[5] / "Arreglos en Gráficos"
ARREGLOS.mkdir(exist_ok=True)
fig1.savefig(ARREGLOS / "barrido_DE_CD_MoO3_MoO3_elite.png", dpi=200, bbox_inches="tight")
print(f"Guardado: {out1.name}")

# ---------------------------------------------------------------------------
# Figura 2 — Parámetros óptimos
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
fig2.suptitle("Variación de parámetros", fontsize=16, fontweight="bold")

axes[0].plot(targets_cm, thetas, "o-", ms=3, color="#2a78d6", lw=1.2)
axes[0].set_ylabel(r"$\phi_1$ (°)")
axes[0].set_ylim(0, 180)
axes[0].set_yticks([0, 45, 90, 135, 180])
axes[0].grid(True, alpha=0.3)

axes[1].plot(targets_cm, d1s, "o-", ms=3, color="#2a78d6", lw=1.2)
axes[1].set_ylabel(r"$d_1$ (nm)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(targets_cm, d2s, "o-", ms=3, color="#2a78d6", lw=1.2)
axes[2].set_ylabel(r"$d_2$ (nm)")
axes[2].grid(True, alpha=0.3)

axes[3].plot(targets_cm, foms, "o-", ms=3, color="#2a78d6", lw=1.2)
axes[3].set_ylabel(r"$f_{\mathrm{obj}}$")
axes[3].set_xlabel(r"$\omega_{\mathrm{obj}}$ (cm$^{-1}$)")
axes[3].grid(True, alpha=0.3)

fig2.tight_layout(rect=(0, 0, 1, 0.965))

out2 = RESULTS_DIR / "params_sweep_elite.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
fig2.savefig(ARREGLOS / "params_DE_CD_MoO3_MoO3_elite.png", dpi=200, bbox_inches="tight")
fig2.savefig(ARREGLOS / "params_DE_CD_MoO3_MoO3_elite.pdf", bbox_inches="tight")
print(f"Guardado: {out2.name}")

plt.show()
