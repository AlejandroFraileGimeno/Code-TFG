# -*- coding: utf-8 -*-
"""
Barrido DE — MgTeMoO6/MoO3
Para cada lambda_target en [FREQ_MIN, FREQ_MAX] (paso FREQ_STEP),
corre Differential Evolution con la NN como surrogate y guarda el resultado.

Checkpointing: si target_XXX.npz ya existe, lo salta.
Relanzar el script retoma desde donde se quedó.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from tensorflow.keras import models as tf_models

ROOT_PATH = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(ROOT_PATH / "Surrogates" / "CD" / "MgTeMoO6_MoO3"))

from generalized_transfer_matrix_method import (
    Air, Au, MgTeMoO6, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf


def _make_fast_predict(model):
    """Inferencia batched directa: mismo calculo que model.predict() pero en
    una sola pasada del grafo, sin trocear en mini-batches de 32 ni pagar el
    overhead por llamada de Keras."""
    import tensorflow as tf
    infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

    def _predict(params_norm):
        return infer(tf.convert_to_tensor(params_norm, tf.float32)).numpy()

    return _predict


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NUM_SEEDS  = 1      # seeds del ensemble
C1         = 1.0    # peso CD_norm
C2         = 0.1    # peso R_total
MAXITER    = 50     # generaciones DE
POPSIZE    = 5      # individuos por parámetro
TOL        = 1e-3   # tolerancia DE
WINDOW_CM  = 10.0   # ventana de búsqueda del pico (cm-1)
BOUNDS = [          # [theta (deg), d1 (nm), d2 (nm)]
    (0,   180),
    (200, 2000),
    (200, 2000),
]
FREQ_STEP = 5       # cm-1
# ---------------------------------------------------------------------------

_CD_MODEL_DIR = ROOT_PATH / "Models" / "CD"      / "MgTeMoO6_MoO3"
_RT_MODEL_DIR = ROOT_PATH / "Models" / "R_total" / "MgTeMoO6_MoO3"
_DATABASE     = str(ROOT_PATH / "Datasets" / "CD" / "MgTeMoO6_MoO3")

_scaler_ref = json.load(open(_CD_MODEL_DIR / "Model_1seed" / "scalers.json"))
FREQ_MIN  = _scaler_ref["feature_min"][3]
FREQ_MAX  = _scaler_ref["feature_max"][3]
N_FREQS   = _scaler_ref["n_freqs"]

FREQS       = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Cargando {NUM_SEEDS} modelos CD_norm...")
_models_cd, _scalers_cd = [], []
for i in range(1, NUM_SEEDS + 1):
    _models_cd.append(_make_fast_predict(tf_models.load_model(
        _CD_MODEL_DIR / f"Model_{i}seed" / f"Model_{i}seed.h5", compile=False
    )))
    _scalers_cd.append(auxf.load_scalers(_CD_MODEL_DIR / f"Model_{i}seed" / "scalers.json"))

print(f"Cargando {NUM_SEEDS} modelos R_total...")
_models_rt, _scalers_rt = [], []
for i in range(1, NUM_SEEDS + 1):
    _models_rt.append(_make_fast_predict(tf_models.load_model(
        _RT_MODEL_DIR / f"Model_{i}seed" / f"Model_{i}seed.h5", compile=False
    )))
    _scalers_rt.append(auxf.load_scalers(_RT_MODEL_DIR / f"Model_{i}seed" / "scalers.json"))

print("Modelos cargados.\n")


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------
def _predict_ensemble(models_list, scalers_list, theta, d1, d2):
    params_batch = np.column_stack([
        np.full(len(FREQS), theta),
        np.full(len(FREQS), d1),
        np.full(len(FREQS), d2),
        FREQS,
    ])
    preds = []
    for predict_fn, sc in zip(models_list, scalers_list):
        params_norm = auxf.normalize_data(params_batch, sc["feature_min"], sc["feature_max"])
        out_std     = predict_fn(params_norm)
        out_norm    = auxf.unstandardize_data(out_std, sc["y_mean"], sc["y_std"])
        out         = auxf.unnormalize_data(out_norm, sc["cd_min"], sc["cd_max"])
        preds.append(np.abs(np.asarray(out, dtype=float).ravel()))
    return np.mean(preds, axis=0)

def _predict_cd(theta, d1, d2):
    return _predict_ensemble(_models_cd, _scalers_cd, theta, d1, d2)

def _predict_r_total(theta, d1, d2):
    return _predict_ensemble(_models_rt, _scalers_rt, theta, d1, d2)


# ---------------------------------------------------------------------------
# FoM
# ---------------------------------------------------------------------------
def _analizar_cd(freqs, cd, target_freq, window):
    freqs = np.asarray(freqs, dtype=float)
    cd    = np.asarray(cd,    dtype=float)

    dist_weight  = 1.0
    miss_penalty = 0.5
    height_rel   = 0.1

    if target_freq is not None and window:
        mask = (freqs > target_freq - window / 2) & (freqs < target_freq + window / 2)
        freqs_w, cd_w = (freqs[mask], cd[mask]) if np.any(mask) else (freqs, cd)
    else:
        freqs_w, cd_w = freqs, cd

    height = height_rel * np.max(cd_w) if np.max(cd_w) > 0 else 0.0
    peaks_w, _ = find_peaks(cd_w, height=height)

    if len(peaks_w) == 0:
        main_peak = int(np.argmin(np.abs(freqs - target_freq))) if target_freq is not None else int(np.argmax(cd))
        f_peak    = freqs[main_peak]
        cd_peak   = cd[main_peak]
        if target_freq is not None:
            norm_dist = abs(f_peak - target_freq) / (window if window else (freqs[-1] - freqs[0]))
            fom = cd_peak - dist_weight * norm_dist - miss_penalty
        else:
            fom = cd_peak
        return {"f_peak": f_peak, "CD_peak": cd_peak, "FoM": fom}

    if target_freq is not None:
        dists      = np.abs(freqs_w[peaks_w] - target_freq)
        candidates = peaks_w[dists == np.min(dists)]
        idx_local  = candidates[np.argmax(cd_w[candidates])] if len(candidates) > 1 else candidates[0]
        main_peak  = int(idx_local) if (freqs_w is freqs) else int(np.where(mask)[0][idx_local])
    else:
        main_peak = int(peaks_w[np.argmax(cd_w[peaks_w])])

    f_peak  = freqs[main_peak]
    cd_peak = cd[main_peak]
    if target_freq is not None:
        norm_dist = abs(f_peak - target_freq) / (window if window else (freqs[-1] - freqs[0]))
        fom = cd_peak - norm_dist
    else:
        fom = cd_peak

    return {"f_peak": f_peak, "CD_peak": cd_peak, "FoM": fom}


def _make_objective(target_freq_cm):
    def objective(params):
        theta, d1, d2 = params
        cd      = _predict_cd(theta, d1, d2)
        info    = _analizar_cd(FREQS, cd, target_freq_cm, WINDOW_CM)
        r_total = _predict_r_total(theta, d1, d2)
        idx     = int(np.argmin(np.abs(FREQS - info["f_peak"])))
        fom     = C1 * info["FoM"] + C2 * float(r_total[idx])
        return -fom
    return objective


def _run_tmm(theta, d1, d2):
    structure = LayeredStructure(
        superstrate=Air(),
        substrate=Au(),
        layers=[
            MgTeMoO6(d=d1 * 1e-9, phi=np.deg2rad(theta)),
            MoO3(d=d2 * 1e-9),
        ],
    )
    results     = [calculate_circular_dichroism_ref(f, 0, structure) for f in FREQS]
    CD_tmm      = np.array([abs(r[1]) for r in results])
    R_total_tmm = np.array([r[2]     for r in results])
    return CD_tmm, R_total_tmm


# ---------------------------------------------------------------------------
# Barrido
# ---------------------------------------------------------------------------
targets_cm = np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP, FREQ_STEP)
n_total    = len(targets_cm)

print(f"Barrido: {n_total} targets  ({FREQ_MIN:.0f}-{FREQ_MAX:.0f} cm-1, paso {FREQ_STEP} cm-1)")
print(f"Resultados en: {RESULTS_DIR}\n")

t_start = time.time()

for idx_t, target_cm in enumerate(targets_cm):
    fname = RESULTS_DIR / f"target_{target_cm:.0f}.npz"

    if fname.exists():
        print(f"[{idx_t+1:3d}/{n_total}]  {target_cm:.0f} cm-1  -> ya existe, saltando")
        continue

    t0 = time.time()
    result = differential_evolution(
        _make_objective(target_cm),
        bounds=BOUNDS,
        strategy="best1bin",
        maxiter=MAXITER,
        popsize=POPSIZE,
        tol=TOL,
        seed=42,
        workers=1,
        polish=False,
    )

    theta_best, d1_best, d2_best = result.x
    fom_nn = -result.fun

    CD_tmm, R_total_tmm = _run_tmm(theta_best, d1_best, d2_best)
    CD_nn      = _predict_cd(theta_best, d1_best, d2_best)
    R_total_nn = _predict_r_total(theta_best, d1_best, d2_best)

    np.savez_compressed(
        fname,
        target_cm   = np.float64(target_cm),
        freqs       = FREQS,
        theta       = np.float64(theta_best),
        d1          = np.float64(d1_best),
        d2          = np.float64(d2_best),
        fom_nn      = np.float64(fom_nn),
        CD_tmm      = CD_tmm,
        R_total_tmm = R_total_tmm,
        CD_nn       = CD_nn,
        R_total_nn  = R_total_nn,
    )

    elapsed       = time.time() - t0
    total_elapsed = time.time() - t_start
    remaining     = total_elapsed / (idx_t + 1) * (n_total - idx_t - 1)
    print(
        f"[{idx_t+1:3d}/{n_total}]  {target_cm:.0f} cm-1  "
        f"th={theta_best:.1f}  d1={d1_best:.0f}  d2={d2_best:.0f} nm  "
        f"FoM={fom_nn:.4f}  "
        f"({elapsed:.0f}s  ~{remaining/60:.1f} min restantes)"
    )

print(f"\nBarrido completado en {(time.time()-t_start)/60:.1f} min")
print(f"Archivos en: {RESULTS_DIR}")
print("Ejecuta plot_sweep.py para ver las graficas.")
