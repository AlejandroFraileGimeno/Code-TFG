# -*- coding: utf-8 -*-
"""
Evaluacion numerica del surrogate forward T_xx — MgTeMoO6/MgTeMoO6/BaF2
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MgTeMoO6, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
NUM_SEEDS  = 3
N_EVAL     = 100
D_MIN      = 200
D_MAX      = 1200
SEED_EVAL  = 99
# ============================================================

MODELS_DIR  = ROOT_PATH / "Models"   / "T_xx" / "MgTeMoO6_MgTeMoO6_BaF2" / "Forward"
DATASET_DIR = ROOT_PATH / "Datasets" / "T_xx" / "MgTeMoO6_MgTeMoO6_BaF2"

scalers   = json.loads((MODELS_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
FREQ_MIN  = scalers["freq_min"]
FREQ_MAX  = scalers["freq_max"]
N_FREQS   = scalers["n_freqs"]
FREQS     = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

print(f"Cargando {NUM_SEEDS} modelo(s) forward...")
models_list = [
    tf.keras.models.load_model(MODELS_DIR / f"Model_{i}seed" / "forward.keras", compile=False)
    for i in range(1, NUM_SEEDS + 1)
]
print("Modelos cargados.\n")

def predict_ensemble(theta1, theta2, d1, d2):
    params = np.array([[theta1, theta2, d1, d2]], dtype=np.float32)
    params_norm = (params - param_min) / (param_max - param_min)
    preds = [m.predict(params_norm, verbose=0)[0] for m in models_list]
    return np.mean(preds, axis=0)

def tmm_spectrum(theta1, theta2, d1, d2):
    structure = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[
            MgTeMoO6(d=d1 * 1e-9, phi=np.deg2rad(theta1)),
            MgTeMoO6(d=d2 * 1e-9, phi=np.deg2rad(theta2)),
        ],
    )
    return np.array([float(calculate_transmission(f, 0, structure, basis="linear")[0]) for f in FREQS])

np.random.seed(SEED_EVAL)
theta1_arr = np.random.uniform(0,     180,   N_EVAL)
theta2_arr = np.random.uniform(0,     180,   N_EVAL)
d1_arr     = np.random.uniform(D_MIN, D_MAX, N_EVAL)
d2_arr     = np.random.uniform(D_MIN, D_MAX, N_EVAL)

r2_list, mae_list, rmse_list = [], [], []
t_start = time.time()

for i in range(N_EVAL):
    th1, th2 = theta1_arr[i], theta2_arr[i]
    d1,  d2  = d1_arr[i],  d2_arr[i]

    T_tmm = tmm_spectrum(th1, th2, d1, d2)
    T_nn  = predict_ensemble(th1, th2, d1, d2)

    ss_res = np.sum((T_tmm - T_nn) ** 2)
    ss_tot = np.sum((T_tmm - T_tmm.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 1e-10 else None
    mae  = float(np.mean(np.abs(T_tmm - T_nn)))
    rmse = float(np.sqrt(np.mean((T_tmm - T_nn) ** 2)))

    if r2 is not None:
        r2_list.append(r2)
    mae_list.append(mae)
    rmse_list.append(rmse)

    r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
    print(f"[{i+1:3d}/{N_EVAL}]  th1={th1:5.1f}  th2={th2:5.1f}  "
          f"d1={d1:6.0f}  d2={d2:6.0f}  |  R2={r2_str}  MAE={mae:.5f}  RMSE={rmse:.5f}")

elapsed = time.time() - t_start
print(f"\n--- Resultados sobre {N_EVAL} estructuras ({NUM_SEEDS} modelos, {elapsed:.0f} s) ---")
print(f"R2   = {np.mean(r2_list):.4f}  +-  {np.std(r2_list):.4f}")
print(f"MAE  = {np.mean(mae_list):.5f}  +-  {np.std(mae_list):.5f}")
print(f"RMSE = {np.mean(rmse_list):.5f}  +-  {np.std(rmse_list):.5f}")