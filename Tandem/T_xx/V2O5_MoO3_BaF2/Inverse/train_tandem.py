# -*- coding: utf-8 -*-
"""
Entrenamiento tandem — red inversa T_xx  V2O5/MoO3/BaF2
=========================================================
Arquitectura tandem:

  T_xx_target -> [Inverse NN] -> params_norm -> [Forward NN, FROZEN] -> T_xx_pred

El loss se calcula entre T_xx_pred y T_xx_target en espacio espectral,
evitando la no-unicidad del problema inverso.

Al terminar guarda en Models/T_xx/V2O5_MoO3_BaF2/Inverse/:
  inverse.keras
  scalers.json  (copia de Forward/scalers.json — misma normalizacion)
  history_loss.txt
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Forward"))

from inverse_model import build_inverse
from forward_model import build_forward

# ============================================================
# CONFIG
# ============================================================
FORWARD_SEED  = 1
N_TRAIN       = 20_000
N_VAL         = 20_000
EPOCHS        = 500
BATCH_SIZE    = 256
LR            = 5e-4
PATIENCE      = 40
NUM_SEEDS     = 1
# ============================================================

DATASET_DIR   = ROOT_PATH / "Datasets" / "T_xx" / "V2O5_MoO3_BaF2"
FORWARD_DIR   = ROOT_PATH / "Models"   / "T_xx" / "V2O5_MoO3_BaF2" / "Forward"
INVERSE_DIR   = ROOT_PATH / "Models"   / "T_xx" / "V2O5_MoO3_BaF2" / f"Inverse_N{N_TRAIN}"
INVERSE_DIR.mkdir(parents=True, exist_ok=True)

SEED_FILE = ROOT_PATH / "Seed" / "SEED_LIST.csv"
seeds = np.loadtxt(SEED_FILE, dtype=int) if SEED_FILE.exists() else np.arange(100)

# ---------------------------------------------------------------------------
# Cargar scalers y dataset
# ---------------------------------------------------------------------------
scalers   = json.loads((FORWARD_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
N_FREQS   = scalers["n_freqs"]

(INVERSE_DIR / "scalers.json").write_text(json.dumps(scalers, indent=2))

print("Cargando dataset...")
params       = np.loadtxt(DATASET_DIR / "params.csv",       delimiter=",", skiprows=1).astype(np.float32)
T_xx_spectra = np.loadtxt(DATASET_DIR / "T_xx_spectra.csv", delimiter=",").astype(np.float32)
N = len(params)
print(f"  {N} muestras  |  N_FREQS={N_FREQS}")

if N_TRAIN + N_VAL > N:
    raise ValueError(f"Dataset insuficiente: {N} < {N_TRAIN + N_VAL}")

params_norm = (params - param_min) / (param_max - param_min)

idx    = np.random.permutation(N)
idx_tr = idx[:N_TRAIN]
idx_va = idx[N_TRAIN:N_TRAIN + N_VAL]

X_tr, X_va = T_xx_spectra[idx_tr], T_xx_spectra[idx_va]
Y_tr, Y_va = params_norm[idx_tr],  params_norm[idx_va]

print(f"  Train: {X_tr.shape}  Val: {X_va.shape}\n")

# ---------------------------------------------------------------------------
# Cargar red forward CONGELADA
# ---------------------------------------------------------------------------
print(f"Cargando forward NN (seed {FORWARD_SEED}, frozen)...")
forward_model = tf.keras.models.load_model(
    FORWARD_DIR / f"Model_{FORWARD_SEED}seed" / "forward.keras", compile=False
)
forward_model.trainable = False
print("Forward NN cargada y congelada.\n")

# ---------------------------------------------------------------------------
# Construccion del grafo tandem
# ---------------------------------------------------------------------------
def build_tandem(inverse_model, forward_model):
    inp     = tf.keras.Input(shape=(N_FREQS,), name="T_xx_target")
    params  = inverse_model(inp)
    T_pred  = forward_model(params)
    return tf.keras.Model(inputs=inp, outputs=T_pred, name="tandem")

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
for i in range(NUM_SEEDS):
    seed = int(seeds[i + 10])
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"=== Inversa {i+1}/{NUM_SEEDS}  (seed={seed}) ===")

    inverse_model = build_inverse(N_FREQS)
    tandem        = build_tandem(inverse_model, forward_model)

    tandem.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    folder = INVERSE_DIR / f"Model_{i+1}seed"
    folder.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            min_delta=1e-7, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6,
        ),
    ]

    print("  Entrenamiento tandem puro (sin warm-up)...")
    t0 = time.time()
    history = tandem.fit(
        X_tr, X_tr,
        validation_data=(X_va, X_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"  Entrenado en {elapsed/60:.1f} min")

    inverse_model.save(folder / "inverse.keras")

    loss_tr  = history.history["loss"]
    loss_val = history.history["val_loss"]
    lines = [f"{e} {loss_val[e]:.8f} {loss_tr[e]:.8f}\n" for e in range(len(loss_tr))]
    (folder / "history_loss.txt").write_text("".join(lines))

    (folder / "hyperparameters.txt").write_text(
        f"n_train={N_TRAIN}\nn_val={N_VAL}\nepochs_run={len(loss_tr)}\n"
        f"batch_size={BATCH_SIZE}\nlr={LR}\npatience={PATIENCE}\n"
        f"loss=MSE_espectral_tandem\nforward_seed={FORWARD_SEED}\nseed={seed}"
    )
    print(f"  Guardado en {folder}\n")

print("Entrenamiento tandem completado.")