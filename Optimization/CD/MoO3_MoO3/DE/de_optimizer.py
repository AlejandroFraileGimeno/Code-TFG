# -*- coding: utf-8 -*-
"""
===========================================================
Differential Evolution — CD optimizer (NN surrogate)
===========================================================
Optimiza la estructura MoO3/MoO3 para maximizar el pico de
dicroísmo circular en una longitud de onda objetivo,
usando dos ensembles de redes neuronales como modelos directos:
  - CD_norm  : (R_r - R_l) / (R_r + R_l)
  - R_total  : R_r + R_l

FoM = C1 × CD_norm_peak + C2 × R_total_at_peak   (C1 >> C2)

Parámetros libres : theta (°), d1 (nm), d2 (nm)
Estructura        : Air / MoO3(d1, phi=theta) / MoO3(d2) / Au

Al final verifica el óptimo con TMM y genera la gráfica.

Author: [Lucia F. Alvarez-Tomillo / Alejandro Fraile]
Date:   [xx/xx/2026]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks, peak_widths
from tensorflow.keras import models as tf_models

# ---------------------------------------------------------------------------
# CONFIG — edita aquí
# ---------------------------------------------------------------------------
TARGET_WAVELENGTH_UM = 15.4    # µm  (longitud de onda objetivo)
WINDOW_CM            = 50.0    # cm⁻¹ (ventana de búsqueda del pico)
NUM_SEEDS            = 5       # modelos del ensemble
C1                   = 1.0    # peso de CD_norm  (dominante)
C2                   = 0.1    # peso de R_total  (secundario)
MAXITER              = 200     # generaciones máximas del DE
POPSIZE              = 15      # individuos por parámetro (población = 15 × 3 = 45)
BOUNDS = [                     # [theta (°), d1 (nm), d2 (nm)]
    (0,   180),
    (200, 2000),
    (200, 2000),
]
# ---------------------------------------------------------------------------

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(ROOT_PATH / "Surrogates" / "CD" / "MoO3, MoO3"))

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf

# ---------------------------------------------------------------------------
# Cargar ensemble de modelos (una sola vez)
# ---------------------------------------------------------------------------
FREQS          = np.linspace(600, 1100, 1000)                                        # cm⁻¹
TARGET_FREQ_CM = (1e4 / TARGET_WAVELENGTH_UM) if TARGET_WAVELENGTH_UM else None      # cm⁻¹

_CD_MODEL_DIR = ROOT_PATH / "Models" / "CD"      / "MoO3_MoO3"
_RT_MODEL_DIR = ROOT_PATH / "Models" / "R_total" / "MoO3_MoO3"
_DATABASE     = str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MoO3")

print(f"Cargando {NUM_SEEDS} modelos CD_norm...")
_models_cd, _scalers_cd = [], []
for i in range(1, NUM_SEEDS + 1):
    _models_cd.append(tf_models.load_model(
        _CD_MODEL_DIR / f"Model_{i}seed" / f"Model_{i}seed.h5", compile=False
    ))
    _scalers_cd.append(str(_CD_MODEL_DIR / f"Model_{i}seed" / "scalers.json"))

print(f"Cargando {NUM_SEEDS} modelos R_total...")
_models_rt, _scalers_rt = [], []
for i in range(1, NUM_SEEDS + 1):
    _models_rt.append(tf_models.load_model(
        _RT_MODEL_DIR / f"Model_{i}seed" / f"Model_{i}seed.h5", compile=False
    ))
    _scalers_rt.append(str(_RT_MODEL_DIR / f"Model_{i}seed" / "scalers.json"))

print(f"Listo. Target: {TARGET_WAVELENGTH_UM} um  ({TARGET_FREQ_CM:.1f} cm-1)\n")


# ---------------------------------------------------------------------------
# Predicción de los ensembles
# ---------------------------------------------------------------------------
def _predict_ensemble(models_list, scalers_list, theta, d1, d2):
    params_batch = np.column_stack([
        np.full(len(FREQS), theta),
        np.full(len(FREQS), d1),
        np.full(len(FREQS), d2),
        FREQS,
    ])
    preds = []
    for m, sp in zip(models_list, scalers_list):
        batch, _ = auxf.predict(m, params_batch, _DATABASE, scaler_path=sp)
        preds.append([abs(float(np.squeeze(v))) for v in batch])
    return np.mean(preds, axis=0)

def _predict_cd(theta, d1, d2):
    return _predict_ensemble(_models_cd, _scalers_cd, theta, d1, d2)

def _predict_r_total(theta, d1, d2):
    return _predict_ensemble(_models_rt, _scalers_rt, theta, d1, d2)


# ---------------------------------------------------------------------------
# Figura de mérito — igual que analizar_CD en target_functions.py
# ---------------------------------------------------------------------------
def _analizar_cd(freqs, cd, target_freq, window):
    freqs = np.asarray(freqs, dtype=float)
    cd    = np.asarray(cd,    dtype=float)

    dist_weight = 1.0
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
        f_peak  = freqs[main_peak]
        cd_peak = cd[main_peak]
        if target_freq is not None:
            norm_dist = abs(f_peak - target_freq) / (window if window else (freqs[-1] - freqs[0]))
            fom = cd_peak - dist_weight * norm_dist - miss_penalty
        else:
            fom = cd_peak
        return {"f_peak": f_peak, "CD_peak": cd_peak, "width": 0.0, "FoM": fom}

    if target_freq is not None:
        dists = np.abs(freqs_w[peaks_w] - target_freq)
        candidates = peaks_w[dists == np.min(dists)]
        idx_local = candidates[np.argmax(cd_w[candidates])] if len(candidates) > 1 else candidates[0]
        main_peak = int(idx_local) if (freqs_w is freqs) else int(np.where(mask)[0][idx_local])
    else:
        main_peak = int(peaks_w[np.argmax(cd_w[peaks_w])])

    try:
        width = peak_widths(cd, [main_peak], rel_height=0.5)[0][0] * (freqs[1] - freqs[0])
    except Exception:
        width = 0.0

    f_peak  = freqs[main_peak]
    cd_peak = cd[main_peak]
    if target_freq is not None:
        norm_dist = abs(f_peak - target_freq) / (window if window else (freqs[-1] - freqs[0]))
        fom = cd_peak - norm_dist
    else:
        fom = cd_peak

    return {"f_peak": f_peak, "CD_peak": cd_peak, "width": width, "FoM": fom}


# ---------------------------------------------------------------------------
# Función objetivo
# ---------------------------------------------------------------------------
_eval_count = [0]

def _combined_fom(theta, d1, d2):
    cd        = _predict_cd(theta, d1, d2)
    info      = _analizar_cd(FREQS, cd, TARGET_FREQ_CM, WINDOW_CM)
    r_total   = _predict_r_total(theta, d1, d2)
    idx_peak  = int(np.argmin(np.abs(FREQS - info["f_peak"])))
    r_at_peak = float(r_total[idx_peak])
    fom       = C1 * info["FoM"] + C2 * r_at_peak
    return fom, info, r_at_peak


def objective(params):
    theta, d1, d2 = params
    _eval_count[0] += 1
    fom, _, _ = _combined_fom(theta, d1, d2)
    return -fom   # DE minimiza → maximizamos FoM


def _callback(xk, convergence):
    theta, d1, d2 = xk
    fom, info, r_at_peak = _combined_fom(theta, d1, d2)
    lambda_peak = 1e4 / info["f_peak"] if info["f_peak"] else float("nan")
    print(
        f"  eval={_eval_count[0]:5d} | FoM={fom:.4f} | "
        f"CD={info['CD_peak']:.3f}  R={r_at_peak:.3f} | "
        f"lam_pico={lambda_peak:.2f} um | "
        f"th={theta:.1f}deg  d1={d1:.0f}  d2={d2:.0f} nm | conv={convergence:.2e}"
    )


# ---------------------------------------------------------------------------
# Optimización
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Differential Evolution (surrogate NN)")
print("=" * 60)

result = differential_evolution(
    objective,
    bounds=BOUNDS,
    strategy="best1bin",
    maxiter=MAXITER,
    popsize=POPSIZE,
    tol=1e-6,
    seed=42,
    callback=_callback,
    workers=1,
    polish=True,
)

theta_best, d1_best, d2_best = result.x

print(f"\n{'='*60}")
print("  Resultado")
print(f"{'='*60}")
print(f"  th = {theta_best:.2f} deg")
print(f"  d1 = {d1_best:.1f} nm")
print(f"  d2 = {d2_best:.1f} nm")
print(f"  FoM (NN) = {-result.fun:.4f}   nfev = {result.nfev}")

# ---------------------------------------------------------------------------
# Verificación con TMM
# ---------------------------------------------------------------------------
print("\nVerificando con TMM...")
structure = LayeredStructure(
    superstrate=Air(),
    substrate=Au(),
    layers=[
        MoO3(d=d1_best * 1e-9, phi=np.deg2rad(theta_best)),
        MoO3(d=d2_best * 1e-9),
    ],
)
tmm_results  = [calculate_circular_dichroism_ref(f, 0, structure) for f in FREQS]
CD_tmm       = np.array([abs(r[1]) for r in tmm_results])
R_total_tmm  = np.array([r[2]     for r in tmm_results])
CD_nn        = _predict_cd(theta_best, d1_best, d2_best)
R_total_nn   = _predict_r_total(theta_best, d1_best, d2_best)

lambda_mu    = 1e4 / FREQS
idx_tmm_best = int(np.argmax(CD_tmm))
r_at_best    = R_total_tmm[idx_tmm_best]
print(f"  TMM:  CD_max = {CD_tmm[idx_tmm_best]:.4f}  R_total = {r_at_best:.4f}  FoM = {CD_tmm[idx_tmm_best]*r_at_best:.4f}  @  lam = {lambda_mu[idx_tmm_best]:.3f} um")
print(f"  NN:   CD_max = {np.max(CD_nn):.4f}  R_total = {R_total_nn[idx_tmm_best]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

ax1.scatter(lambda_mu, CD_tmm,      s=5, c="red",    label="CD TMM")
ax1.scatter(lambda_mu, CD_nn,       s=5, c="blue",   label="CD NN")
ax2.scatter(lambda_mu, R_total_tmm, s=5, c="red",    label="R_total TMM")
ax2.scatter(lambda_mu, R_total_nn,  s=5, c="blue",   label="R_total NN")

for ax in (ax1, ax2):
    if TARGET_WAVELENGTH_UM is not None:
        ax.axvline(TARGET_WAVELENGTH_UM, color="gray", ls="--", lw=1,
                   label=f"target  {TARGET_WAVELENGTH_UM} µm")
    ax.legend()

ax1.set_ylabel("CD reflection (normalizado)")
ax2.set_ylabel("R_total = R_r + R_l")
ax2.set_xlabel(r"$\lambda$ (µm)")
fig.suptitle(
    f"DE óptimo — θ={theta_best:.1f}°   d1={d1_best:.0f} nm   d2={d2_best:.0f} nm\n"
    f"CD={CD_tmm[idx_tmm_best]:.4f}  R={r_at_best:.4f}  FoM={CD_tmm[idx_tmm_best]*r_at_best:.4f}  "
    f"@  λ={lambda_mu[idx_tmm_best]:.3f} µm"
)
fig.tight_layout()

save_path = results_dir / f"best_th{theta_best:.0f}_d1{d1_best:.0f}_d2{d2_best:.0f}.png"
fig.savefig(save_path, dpi=150)
print(f"\nGráfica guardada en: {save_path.relative_to(ROOT_PATH)}")
plt.show()