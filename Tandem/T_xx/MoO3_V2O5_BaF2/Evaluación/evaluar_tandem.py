# -*- coding: utf-8 -*-
"""
Evaluación numérica de la red inversa (tandem) — T_xx  MoO3/V2O5/BaF2
=======================================================================
Para N espectros del conjunto de test:
  1. Inversa NN → (θ₁, θ₂, d₁, d₂)
  2. TMM con esos parámetros → T_xx_tmm
  3. Compara T_xx_tmm con el espectro objetivo
  Métricas: MAE, RMSE, R² entre objetivo y TMM reconstruido
"""

import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, V2O5, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
NUM_SEEDS  = 1
N_EVAL     = 200    # espectros de test a evaluar
SEED_EVAL  = 123
# ============================================================

INVERSE_DIR = ROOT_PATH / "Models" / "T_xx" / "MoO3_V2O5_BaF2" / "Inverse"
DATASET_DIR = ROOT_PATH / "Datasets" / "T_xx" / "MoO3_V2O5_BaF2"

scalers   = json.loads((INVERSE_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
N_FREQS   = scalers["n_freqs"]
FREQS     = np.linspace(scalers["freq_min"], scalers["freq_max"], N_FREQS)

print(f"Cargando {NUM_SEEDS} modelo(s) inverso(s)...")
inv_models = [
    tf.keras.models.load_model(INVERSE_DIR / f"Model_{i}seed" / "inverse.keras", compile=False)
    for i in range(1, NUM_SEEDS + 1)
]
print("Modelos cargados.\n")

# Cargar dataset (últimas muestras → fuera del train+val)
print("Cargando dataset...")
params_all   = np.loadtxt(DATASET_DIR / "params.csv",       delimiter=",", skiprows=1).astype(np.float32)
T_xx_all     = np.loadtxt(DATASET_DIR / "T_xx_spectra.csv", delimiter=",").astype(np.float32)
N_TOTAL      = len(params_all)
N_USED       = 400_000 + 50_000   # train + val usados en entrenamiento
test_idx     = np.arange(N_USED, N_TOTAL)
np.random.seed(SEED_EVAL)
chosen       = np.random.choice(test_idx, size=min(N_EVAL, len(test_idx)), replace=False)
T_targets    = T_xx_all[chosen]
print(f"  {len(chosen)} espectros de test (índices {chosen.min()}–{chosen.max()})\n")

def inverse_ensemble(spectrum):
    inp  = spectrum.reshape(1, -1).astype(np.float32)
    preds = [m.predict(inp, verbose=0)[0] for m in inv_models]
    return np.mean(preds, axis=0)   # params normalizados, shape (4,)

def denorm(params_norm):
    return params_norm * (param_max - param_min) + param_min

def tmm_spectrum(theta1, theta2, d1, d2):
    structure = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[
            MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta1)),
            V2O5(d=d2 * 1e-9, phi=np.deg2rad(theta2)),
        ],
    )
    return np.array([float(calculate_transmission(f, 0, structure, basis="linear")[0])
                     for f in FREQS])

mae_list, rmse_list, r2_list = [], [], []

for k, idx in enumerate(chosen):
    target   = T_targets[k]
    p_norm   = inverse_ensemble(target)
    th1, th2, d1, d2 = denorm(p_norm)

    T_tmm    = tmm_spectrum(th1, th2, d1, d2)

    ss_res   = np.sum((target - T_tmm) ** 2)
    ss_tot   = np.sum((target - target.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else None
    mae      = float(np.mean(np.abs(target - T_tmm)))
    rmse     = float(np.sqrt(np.mean((target - T_tmm) ** 2)))

    mae_list.append(mae); rmse_list.append(rmse)
    if r2 is not None: r2_list.append(r2)

    r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
    print(f"[{k+1:3d}/{len(chosen)}]  θ₁={th1:5.1f}°  θ₂={th2:5.1f}°  "
          f"d₁={d1:5.0f}  d₂={d2:5.0f} nm  |  R²={r2_str}  MAE={mae:.5f}  RMSE={rmse:.5f}")

print(f"\n--- Resumen sobre {len(chosen)} espectros de test ---")
print(f"R²   = {np.mean(r2_list):.4f}  ±  {np.std(r2_list):.4f}")
print(f"MAE  = {np.mean(mae_list):.5f}  ±  {np.std(mae_list):.5f}")
print(f"RMSE = {np.mean(rmse_list):.5f}  ±  {np.std(rmse_list):.5f}")