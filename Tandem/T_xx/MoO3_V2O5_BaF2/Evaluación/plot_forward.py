# -*- coding: utf-8 -*-
"""
Comparativa visual TMM vs Forward NN — T_xx  MoO3/V2O5/BaF2
=============================================================
Genera N_PRED figuras con el espectro T_xx(f) de TMM y NN superpuestos.
Guarda los PNG en Evaluación/resultados_forward/.
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, V2O5, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
NUM_SEEDS  = 3
N_PRED     = 8      # estructuras a representar
D_MIN      = 200
D_MAX      = 1200
SEED_PLOT  = 77
# ============================================================

MODELS_DIR = ROOT_PATH / "Models" / "T_xx" / "MoO3_V2O5_BaF2" / "Forward"
OUT_DIR    = Path(__file__).parent / "resultados_forward"
OUT_DIR.mkdir(exist_ok=True)

scalers   = json.loads((MODELS_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
FREQ_MIN  = scalers["freq_min"]
FREQ_MAX  = scalers["freq_max"]
N_FREQS   = scalers["n_freqs"]
FREQS     = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

print(f"Cargando {NUM_SEEDS} modelo(s) forward...")
models_list = []
for i in range(1, NUM_SEEDS + 1):
    models_list.append(
        tf.keras.models.load_model(MODELS_DIR / f"Model_{i}seed" / "forward.keras", compile=False)
    )
print("Modelos cargados.\n")

def predict_ensemble(theta1, theta2, d1, d2):
    params = np.array([[theta1, theta2, d1, d2]], dtype=np.float32)
    params_norm = (params - param_min) / (param_max - param_min)
    preds = [m.predict(params_norm, verbose=0)[0] for m in models_list]
    mean = np.mean(preds, axis=0)
    std  = np.std(preds,  axis=0)
    return mean, std

def tmm_spectrum(theta1, theta2, d1, d2):
    structure = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[
            MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta1)),
            V2O5(d=d2 * 1e-9, phi=np.deg2rad(theta2)),
        ],
    )
    return np.array([float(calculate_transmission(f, 0, structure, basis="linear")[0]) for f in FREQS])

# ---------------------------------------------------------------------------
np.random.seed(SEED_PLOT)

ncols = 2
nrows = math.ceil(N_PRED / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
axes = axes.flat

for i in range(N_PRED):
    th1 = np.random.uniform(0,     180)
    th2 = np.random.uniform(0,     180)
    d1  = np.random.uniform(D_MIN, D_MAX)
    d2  = np.random.uniform(D_MIN, D_MAX)

    T_tmm         = tmm_spectrum(th1, th2, d1, d2)
    T_nn, T_nn_std = predict_ensemble(th1, th2, d1, d2)

    mae  = float(np.mean(np.abs(T_tmm - T_nn)))
    rmse = float(np.sqrt(np.mean((T_tmm - T_nn) ** 2)))

    ax = axes[i]
    ax.plot(FREQS, T_tmm, "r-",    lw=1.5, label="TMM")
    ax.plot(FREQS, T_nn,  "b--",   lw=1.5, label="NN (media)")
    ax.fill_between(FREQS, T_nn - T_nn_std, T_nn + T_nn_std,
                    alpha=0.2, color="blue", label="NN ±σ")
    ax.set_xlabel("Número de onda (cm⁻¹)", fontsize=10)
    ax.set_ylabel("T_xx", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"θ₁={th1:.0f}°  θ₂={th2:.0f}°  d₁={d1:.0f}  d₂={d2:.0f} nm\n"
        f"MAE={mae:.4f}  RMSE={rmse:.4f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    print(f"[{i+1}/{N_PRED}]  MAE={mae:.4f}  RMSE={rmse:.4f}")

for j in range(i + 1, nrows * ncols):
    axes[j].set_visible(False)

fig.suptitle("Surrogate forward T_xx  —  TMM vs NN  [MoO3/V2O5/BaF2]", fontsize=12)
fig.tight_layout()

out = OUT_DIR / "comparativa_forward.png"
fig.savefig(out, dpi=150)
print(f"\nGuardado: {out}")
plt.show()