# -*- coding: utf-8 -*-
"""
Visualizacion del barrido DE — V2O5/V2O5
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
    "font.size":           12,
    "axes.labelsize":      13,
    "axes.titlesize":      12,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     11,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

RESULTS_DIR = Path(__file__).parent / "results"

files = sorted(
    RESULTS_DIR.glob("target_*.npz"),
    key=lambda f: float(f.stem.split("_")[1]),
)

if not files:
    print(f"No hay resultados en {RESULTS_DIR}")
    print("Ejecuta sweep.py primero.")
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

print(f"  {len(targets_cm)} targets  |  frecuencias {freqs[0]:.0f}-{freqs[-1]:.0f} cm-1\n")

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
cbar.set_label(r"$|\mathrm{CD}|$ normalizado (TMM)")

f_diag = [max(freqs[0], targets_cm[0]), min(freqs[-1], targets_cm[-1])]
ax1.plot(f_diag, f_diag, "w--", lw=1.2, alpha=0.6, label=r"target = $\omega$")

ax1.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=13)
ax1.set_ylabel(r"$\omega_\mathrm{target}$ (cm$^{-1}$)")
ax1.set_title("Barrido DE - V2O5/V2O5   |CD| optimo", fontsize=13)
ax1.legend(fontsize=10)
fig1.tight_layout()

out1 = RESULTS_DIR / "heatmap_CD.png"
fig1.savefig(out1, dpi=200, bbox_inches="tight")
print(f"Guardado: {out1.name}")

# ---------------------------------------------------------------------------
# Figura 2 — Parametros optimos
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

axes[0].plot(targets_cm, thetas, "o-", ms=3, color="#2a78d6", lw=1.2)
axes[0].set_ylabel(r"$\theta$ (°)")
axes[0].set_ylim(0, 180)
axes[0].set_yticks([0, 45, 90, 135, 180])
axes[0].grid(True, alpha=0.3)

axes[1].plot(targets_cm, d1s, "o-", ms=3, color="#2a78d6", lw=1.2, label="$d_1$ (V2O5, rotada)")
axes[1].set_ylabel(r"$d_1$ (nm)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

axes[2].plot(targets_cm, d2s, "o-", ms=3, color="#2a78d6", lw=1.2, label="$d_2$ (V2O5)")
axes[2].set_ylabel(r"$d_2$ (nm)")
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=10)

axes[3].plot(targets_cm, foms, "o-", ms=3, color="#2a78d6", lw=1.2)
axes[3].set_ylabel("FoM (NN)", fontsize=12)
axes[3].set_xlabel(r"$\omega_\mathrm{target}$ (cm$^{-1}$)")
axes[3].grid(True, alpha=0.3)

fig2.suptitle("Parametros optimos por target - V2O5/V2O5", fontsize=13)
fig2.tight_layout()

out2 = RESULTS_DIR / "params_sweep.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Guardado: {out2.name}")

plt.show()
