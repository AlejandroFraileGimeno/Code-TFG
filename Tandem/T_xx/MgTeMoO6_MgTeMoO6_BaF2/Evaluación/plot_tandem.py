# -*- coding: utf-8 -*-
"""
Comparativa visual: espectro objetivo vs TMM con parametros de la inversa
=========================================================================
Muestra N_PLOT casos del test: objetivo (negro) vs TMM reconstruido (rojo).
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
    Air, BaF2, MgTeMoO6, LayeredStructure, calculate_transmission,
)

NUM_SEEDS = 1
N_PLOT    = 8
SEED_PLOT = 55
N_TRAIN   = 20_000

INVERSE_DIR = ROOT_PATH / "Models"   / "T_xx" / "MgTeMoO6_MgTeMoO6_BaF2" / f"Inverse_N{N_TRAIN}"
DATASET_DIR = ROOT_PATH / "Datasets" / "T_xx" / "MgTeMoO6_MgTeMoO6_BaF2"
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
        layers=[MgTeMoO6(d=d1*1e-9, phi=np.deg2rad(th1)),
                MgTeMoO6(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0]) for f in FREQS])

ncols = 2
nrows = math.ceil(N_PLOT / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
axes = axes.flat

for i, idx in enumerate(chosen):
    target            = T_xx_all[idx]
    th1, th2, d1, d2  = denorm(inverse_ensemble(target))
    T_tmm             = tmm_spectrum(th1, th2, d1, d2)

    mae  = float(np.mean(np.abs(target - T_tmm)))
    rmse = float(np.sqrt(np.mean((target - T_tmm)**2)))

    ax = axes[i]
    ax.plot(FREQS, target, "k-",  lw=2,   label="Objetivo (dataset)")
    ax.plot(FREQS, T_tmm,  "r--", lw=1.5, label="TMM(params inversa)")
    ax.set_xlabel("Numero de onda (cm-1)", fontsize=10)
    ax.set_ylabel("T_xx", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"th1={th1:.0f}  th2={th2:.0f}  d1={d1:.0f}  d2={d2:.0f} nm\n"
        f"MAE={mae:.4f}  RMSE={rmse:.4f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    print(f"[{i+1}/{N_PLOT}]  MAE={mae:.4f}  RMSE={rmse:.4f}")

for j in range(i + 1, nrows * ncols):
    axes[j].set_visible(False)

fig.suptitle("Evaluacion tandem — Objetivo vs TMM(params inversa)  [MgTeMoO6/MgTeMoO6/BaF2]",
             fontsize=12)
fig.tight_layout()
out = OUT_DIR / "comparativa_tandem.png"
fig.savefig(out, dpi=150)
print(f"\nGuardado: {out}")
plt.show()