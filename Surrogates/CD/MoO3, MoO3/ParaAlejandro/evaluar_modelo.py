# -*- coding: utf-8 -*-
"""Evalua el modelo entrenado: compara CD predicho (NN) vs CD real (TMM)."""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from pathlib import Path
import numpy as np
from tensorflow.keras import models

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf

MODEL_DIR = Path("NN_Code/Forward_Models_Trained_bilayers_MoO3/Model_1seed")
# El modelo se guardo con Keras 3; lo cargamos con Keras 3 (su loader h5 usa
# h5py por debajo y maneja bien rutas con tildes).
model = models.load_model(str(MODEL_DIR / "Model_1seed.h5"), compile=False)
scaler_path = MODEL_DIR / "scalers.json"

freqs = np.linspace(600, 1100, 1000)
rng = np.random.default_rng(12345)   # semilla distinta a la del dataset
N_TEST = 8

all_true, all_pred = [], []
print(f"{'theta':>6} {'d1':>5} {'d2':>5} | {'MAE':>10} {'RMSE':>10} {'corr':>6}")
for _ in range(N_TEST):
    d1, d2 = rng.integers(200, 2001, size=2)
    theta = rng.integers(0, 181)
    structure = LayeredStructure(
        superstrate=Air(), substrate=Au(),
        layers=[MoO3(d=d1*1e-9, phi=np.deg2rad(theta)), MoO3(d=d2*1e-9)],
    )
    cd_true = np.array([abs(calculate_circular_dichroism_ref(f, 0, structure)[1]) for f in freqs])
    params = np.column_stack([np.full(1000, theta), np.full(1000, d1),
                              np.full(1000, d2), freqs])
    cd_pred, _ = auxf.predict(model, params, None, scaler_path=scaler_path)
    cd_pred = np.abs(np.squeeze(cd_pred))
    mae = np.mean(np.abs(cd_pred - cd_true))
    rmse = np.sqrt(np.mean((cd_pred - cd_true)**2))
    corr = np.corrcoef(cd_pred, cd_true)[0, 1]
    print(f"{theta:6d} {d1:5d} {d2:5d} | {mae:10.5f} {rmse:10.5f} {corr:6.3f}")
    all_true.append(cd_true); all_pred.append(cd_pred)

all_true = np.concatenate(all_true); all_pred = np.concatenate(all_pred)
print("-"*50)
print(f"GLOBAL  MAE  = {np.mean(np.abs(all_pred-all_true)):.5f}")
print(f"GLOBAL  RMSE = {np.sqrt(np.mean((all_pred-all_true)**2)):.5f}")
print(f"GLOBAL  corr = {np.corrcoef(all_pred, all_true)[0,1]:.3f}")
print(f"Rango CD real: [{all_true.min():.4f}, {all_true.max():.4f}], media {all_true.mean():.4f}")
print(f"MAE relativo a la media del CD: {100*np.mean(np.abs(all_pred-all_true))/all_true.mean():.1f}%")
