# -*- coding: utf-8 -*-
"""
Comparativa visual TMM vs Forward NN — T_xx  V2O5/MoO3/BaF2
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

NUM_SEEDS = 3
N_PLOT    = 8
D_MIN     = 200
D_MAX     = 1200
SEED_PLOT = 77

MODELS_DIR = ROOT_PATH / "Models" / "T_xx" / "V2O5_MoO3_BaF2" / "Forward"
OUT_DIR    = Path(__file__).parent / "resultados_forward"
OUT_DIR.mkdir(exist_ok=True)

scalers   = json.loads((MODELS_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
N_FREQS   = scalers["n_freqs"]
FREQS     = np.linspace(scalers["freq_min"], scalers["freq_max"], N_FREQS)

models_list = [
    tf.keras.models.load_model(MODELS_DIR / f"Model_{i}seed" / "forward.keras", compile=False)
    for i in range(1, NUM_SEEDS + 1)
]

np.random.seed(SEED_PLOT)
theta1_arr = np.random.uniform(0,     180,   N_PLOT)
theta2_arr = np.random.uniform(0,     180,   N_PLOT)
d1_arr     = np.random.uniform(D_MIN, D_MAX, N_PLOT)
d2_arr     = np.random.uniform(D_MIN, D_MAX, N_PLOT)

def predict_ensemble(th1, th2, d1, d2):
    p = np.array([[th1, th2, d1, d2]], dtype=np.float32)
    p_norm = (p - param_min) / (param_max - param_min)
    return np.mean([m.predict(p_norm, verbose=0)[0] for m in models_list], axis=0)

def tmm_spectrum(th1, th2, d1, d2):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[V2O5(d=d1*1e-9, phi=np.deg2rad(th1)),
                MoO3(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0]) for f in FREQS])

ncols = 2
nrows = math.ceil(N_PLOT / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
axes = axes.flat

for i in range(N_PLOT):
    th1, th2 = theta1_arr[i], theta2_arr[i]
    d1,  d2  = d1_arr[i],  d2_arr[i]
    T_tmm = tmm_spectrum(th1, th2, d1, d2)
    T_nn  = predict_ensemble(th1, th2, d1, d2)
    mae   = float(np.mean(np.abs(T_tmm - T_nn)))
    rmse  = float(np.sqrt(np.mean((T_tmm - T_nn)**2)))

    ax = axes[i]
    ax.plot(FREQS, T_tmm, color="#0b0b0b", lw=2.0, label="TMM")
    ax.plot(FREQS, T_nn,  color="#2a78d6", lw=1.6, ls="--", label="NN")
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=10)
    ax.set_ylabel(r"$T_{xx}$", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"th1={th1:.0f}  th2={th2:.0f}  d1={d1:.0f}  d2={d2:.0f} nm\n"
        f"MAE={mae:.4f}  RMSE={rmse:.4f}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True)
    print(f"[{i+1}/{N_PLOT}]  MAE={mae:.4f}  RMSE={rmse:.4f}")

for j in range(i + 1, nrows * ncols):
    axes[j].set_visible(False)

fig.suptitle("Evaluacion forward NN — TMM vs NN  [V2O5/MoO3/BaF2]", fontsize=12)
fig.tight_layout()
out = OUT_DIR / "comparativa_forward.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nGuardado: {out}")
plt.show()