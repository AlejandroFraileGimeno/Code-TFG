# -*- coding: utf-8 -*-
"""
Comparativa visual: espectro objetivo vs TMM con parámetros de la inversa
=========================================================================
Muestra N_PLOT casos del test: objetivo (negro) vs TMM reconstruido (rojo).
Guarda en Evaluación/resultados_tandem/
"""

import sys
import json
import math
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
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, V2O5, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
NUM_SEEDS = 1
N_PLOT    = 8
SEED_PLOT = 55
N_TRAIN   = 20_000
# ============================================================

INVERSE_DIR = ROOT_PATH / "Models"   / "T_xx" / "MoO3_V2O5_BaF2" / f"Inverse_N{N_TRAIN}"
DATASET_DIR = ROOT_PATH / "Datasets" / "T_xx" / "MoO3_V2O5_BaF2"
OUT_DIR     = Path(__file__).parent / "resultados_tandem"
OUT_DIR.mkdir(exist_ok=True)

scalers   = json.loads((INVERSE_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
N_FREQS   = scalers["n_freqs"]
FREQS     = np.linspace(scalers["freq_min"], scalers["freq_max"], N_FREQS)

inv_models = [
    tf.keras.models.load_model(INVERSE_DIR / f"Model_{i}seed" / "inverse.keras", compile=False)
    for i in range(1, NUM_SEEDS + 1)
]

T_xx_all = np.loadtxt(DATASET_DIR / "T_xx_spectra.csv", delimiter=",").astype(np.float32)
N_USED   = N_TRAIN * 2
np.random.seed(SEED_PLOT)
chosen   = np.random.choice(np.arange(N_USED, len(T_xx_all)), size=N_PLOT, replace=False)

def inverse_ensemble(spectrum):
    inp   = spectrum.reshape(1, -1).astype(np.float32)
    preds = [m.predict(inp, verbose=0)[0] for m in inv_models]
    return np.mean(preds, axis=0)

def denorm(p): return p * (param_max - param_min) + param_min

def tmm_spectrum(th1, th2, d1, d2):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[MoO3(d=d1*1e-9, phi=np.deg2rad(th1)),
                V2O5(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0]) for f in FREQS])

ncols = 2
nrows = math.ceil(N_PLOT / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
axes = axes.flat

for i, idx in enumerate(chosen):
    target             = T_xx_all[idx]
    th1, th2, d1, d2   = denorm(inverse_ensemble(target))
    T_tmm              = tmm_spectrum(th1, th2, d1, d2)

    mae  = float(np.mean(np.abs(target - T_tmm)))
    rmse = float(np.sqrt(np.mean((target - T_tmm)**2)))

    ax = axes[i]
    ax.plot(FREQS, target, color="#0b0b0b", lw=2.0, label="Objetivo (dataset)")
    ax.plot(FREQS, T_tmm,  color="#2a78d6", lw=1.6, ls="--", label="TMM (parámetros inversa)")
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=10)
    ax.set_ylabel(r"$T_{xx}$", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"θ₁={th1:.0f}°  θ₂={th2:.0f}°  d₁={d1:.0f}  d₂={d2:.0f} nm\n"
        f"MAE={mae:.4f}  RMSE={rmse:.4f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True)
    print(f"[{i+1}/{N_PLOT}]  MAE={mae:.4f}  RMSE={rmse:.4f}")

for j in range(i + 1, nrows * ncols):
    axes[j].set_visible(False)

fig.suptitle("Evaluación tandem — Objetivo vs TMM(params inversa)  [MoO3/V2O5/BaF2]",
             fontsize=12)
fig.tight_layout()
out = OUT_DIR / "comparativa_tandem.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nGuardado: {out}")
plt.show()