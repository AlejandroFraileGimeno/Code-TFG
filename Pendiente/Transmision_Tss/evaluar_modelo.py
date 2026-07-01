# -*- coding: utf-8 -*-
"""Evalua el modelo T_ss: compara T_ss predicho (red) vs T_ss real (TMM)."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from pathlib import Path
import numpy as np
from tensorflow.keras import models

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, LayeredStructure, calculate_transmission,
)
import utils_nn_forward as auxf

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "NN_Code" / "Tss_Models_Trained_bilayers_MoO3" / "Model_1seed"
model = models.load_model(str(MODEL_DIR / "Model_1seed.h5"), compile=False)
scaler_path = MODEL_DIR / "scalers.json"

freqs = np.linspace(600, 1100, 1000)
rng = np.random.default_rng(12345)   # semilla distinta a la del dataset
N_TEST = 8

all_true, all_pred = [], []
print(f"{'th1':>4} {'th2':>4} {'d1':>5} {'d2':>5} | {'MAE':>10} {'RMSE':>10} {'corr':>6}")
def tmm_tss(d1, d2, theta1, theta2):
    structure = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[MoO3(d=d1*1e-9, phi=np.deg2rad(theta1)),
                MoO3(d=d2*1e-9, phi=np.deg2rad(theta2))],
    )
    return np.array([float(calculate_transmission(f, 0, structure, basis="linear")[1])
                     for f in freqs])


for _ in range(N_TEST):
    # Solo evaluamos sobre estructuras FISICAS (T_ss<=1): si el TMM explota,
    # rechazamos y probamos otra (igual que en la generacion del dataset).
    while True:
        d1, d2 = rng.integers(200, 2001, size=2)
        theta1, theta2 = rng.integers(0, 181, size=2)
        t_true = tmm_tss(d1, d2, theta1, theta2)
        if np.all(np.isfinite(t_true)) and t_true.max() <= 1.0 and t_true.min() >= 0.0:
            break
    # Orden de entradas: theta1, theta2, d1, d2, freq
    params = np.column_stack([np.full(1000, theta1), np.full(1000, theta2),
                              np.full(1000, d1), np.full(1000, d2), freqs])
    t_pred, _ = auxf.predict(model, params, None, scaler_path=scaler_path)
    t_pred = np.abs(np.squeeze(t_pred))
    mae = np.mean(np.abs(t_pred - t_true))
    rmse = np.sqrt(np.mean((t_pred - t_true)**2))
    corr = np.corrcoef(t_pred, t_true)[0, 1]
    print(f"{theta1:4d} {theta2:4d} {d1:5d} {d2:5d} | {mae:10.5f} {rmse:10.5f} {corr:6.3f}")
    all_true.append(t_true); all_pred.append(t_pred)

all_true = np.concatenate(all_true); all_pred = np.concatenate(all_pred)
print("-"*50)
print(f"GLOBAL  MAE  = {np.mean(np.abs(all_pred-all_true)):.5f}")
print(f"GLOBAL  RMSE = {np.sqrt(np.mean((all_pred-all_true)**2)):.5f}")
print(f"GLOBAL  corr = {np.corrcoef(all_pred, all_true)[0,1]:.3f}")
print(f"Rango T_ss real: [{all_true.min():.4f}, {all_true.max():.4f}], media {all_true.mean():.4f}")
print(f"MAE relativo a la media de T_ss: {100*np.mean(np.abs(all_pred-all_true))/all_true.mean():.1f}%")
