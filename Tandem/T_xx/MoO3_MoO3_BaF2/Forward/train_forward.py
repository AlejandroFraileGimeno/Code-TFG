# -*- coding: utf-8 -*-
"""
Entrenamiento de la red forward espectral — T_xx  MoO3/MoO3/BaF2

Dataset esperado en Datasets/T_xx/MoO3_MoO3_BaF2/:
  params.csv        (N, 4)        theta1_deg, theta2_deg, d1_nm, d2_nm
  T_xx_spectra.csv  (N, N_FREQS)  espectros T_xx en [0, 1]
  freqs.csv         (N_FREQS,)    eje de frecuencias en cm-1

Modelo guardado en Models/T_xx/MoO3_MoO3_BaF2/Forward/
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from forward_model import build_forward

# ============================================================
# CONFIG
# ============================================================
N_TRAIN      = 400_000
N_VAL        = 50_000
EPOCHS       = 300
BATCH_SIZE   = 512
LR           = 1e-3
PATIENCE     = 25
NUM_SEEDS    = 1
# ============================================================

DATASET_DIR = ROOT_PATH / "Datasets" / "T_xx" / "MoO3_MoO3_BaF2"
MODELS_DIR  = ROOT_PATH / "Models"   / "T_xx" / "MoO3_MoO3_BaF2" / "Forward"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED_FILE = ROOT_PATH / "Seed" / "SEED_LIST.csv"
seeds = np.loadtxt(SEED_FILE, dtype=int) if SEED_FILE.exists() else np.arange(100)

print("Cargando dataset...")
params       = np.loadtxt(DATASET_DIR / "params.csv",       delimiter=",", skiprows=1).astype(np.float32)
T_xx_spectra = np.loadtxt(DATASET_DIR / "T_xx_spectra.csv", delimiter=",").astype(np.float32)
freqs        = np.loadtxt(DATASET_DIR / "freqs.csv",        delimiter=",")

N, N_FREQS = T_xx_spectra.shape
print(f"  {N} muestras  |  {N_FREQS} frecuencias  |  params shape {params.shape}")

if N_TRAIN + N_VAL > N:
    raise ValueError(f"Dataset tiene {N} muestras pero se piden {N_TRAIN + N_VAL}.")

param_min = params.min(axis=0)
param_max = params.max(axis=0)
params_norm = (params - param_min) / (param_max - param_min)

scalers = {
    "param_min":   param_min.tolist(),
    "param_max":   param_max.tolist(),
    "param_names": ["theta1_deg", "theta2_deg", "d1_nm", "d2_nm"],
    "freq_min":    float(freqs.min()),
    "freq_max":    float(freqs.max()),
    "n_freqs":     int(N_FREQS),
}
(MODELS_DIR / "scalers.json").write_text(json.dumps(scalers, indent=2))
print(f"  scalers.json guardado en {MODELS_DIR}")

idx    = np.random.permutation(N)
idx_tr = idx[:N_TRAIN]
idx_va = idx[N_TRAIN:N_TRAIN + N_VAL]

X_tr, Y_tr = params_norm[idx_tr], T_xx_spectra[idx_tr]
X_va, Y_va = params_norm[idx_va], T_xx_spectra[idx_va]
print(f"  Train: {X_tr.shape}   Val: {X_va.shape}\n")

for i in range(NUM_SEEDS):
    seed = int(seeds[i])
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"=== Modelo {i+1}/{NUM_SEEDS}  (seed={seed}) ===")

    model = build_forward(N_FREQS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae"],
    )

    folder = MODELS_DIR / f"Model_{i+1}seed"
    folder.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            min_delta=1e-6, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"  Entrenado en {elapsed/60:.1f} min")

    model.save(folder / "forward.keras")

    loss_tr  = history.history["loss"]
    loss_val = history.history["val_loss"]
    lines = [f"{e} {loss_val[e]:.8f} {loss_tr[e]:.8f}\n" for e in range(len(loss_tr))]
    (folder / "history_loss.txt").write_text("".join(lines))

    (folder / "hyperparameters.txt").write_text(
        f"n_train={N_TRAIN}\nn_val={N_VAL}\nepochs_run={len(loss_tr)}\n"
        f"batch_size={BATCH_SIZE}\nlr={LR}\npatience={PATIENCE}\n"
        f"loss=MSE\noptimizer=Adam\nseed={seed}"
    )
    print(f"  Guardado en {folder}\n")

print("Entrenamiento forward completado.")