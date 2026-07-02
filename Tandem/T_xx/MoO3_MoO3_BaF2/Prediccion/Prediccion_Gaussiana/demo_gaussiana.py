# -*- coding: utf-8 -*-
"""
Barrido de dips gaussianos -- diseno inverso T_xx  MoO3/MoO3/BaF2
===================================================================
T(f) = 1 - exp( -(f - f0)^2 / (2*sigma^2) )
El ajuste se mide SOLO en la ventana [f0 - K*sigma, f0 + K*sigma].

Salidas:
  resultados/heatmap_r2_<EVAL_MODE>.png
  resultados/ejemplos_buenos.png
  resultados/ejemplos_malos.png
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
N_TRAIN    = 20_000
K          = 3
A          = 1.0

F0_MIN     = 500.0
F0_MAX     = 1300.0
N_F0       = 250

SIGMA_MIN  = 20.0
SIGMA_MAX  = 150.0
N_SIGMA    = 50

FREQ_MIN   = 400.0
FREQ_MAX   = 1400.0

EVAL_MODE  = "forward"
# ============================================================

INVERSE_DIR = ROOT_PATH / "Models" / "T_xx" / "MoO3_MoO3_BaF2" / f"Inverse_N{N_TRAIN}"
FORWARD_DIR = ROOT_PATH / "Models" / "T_xx" / "MoO3_MoO3_BaF2" / "Forward"
OUT_DIR     = Path(__file__).parent / "resultados"
OUT_DIR.mkdir(exist_ok=True)

scalers   = json.loads((INVERSE_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
N_FREQS   = scalers["n_freqs"]
FREQS     = np.linspace(scalers["freq_min"], scalers["freq_max"], N_FREQS)

fwd_scalers   = json.loads((FORWARD_DIR / "scalers.json").read_text())
fwd_param_min = np.array(fwd_scalers["param_min"], dtype=np.float32)
fwd_param_max = np.array(fwd_scalers["param_max"], dtype=np.float32)

print(f"Cargando modelos  (EVAL_MODE='{EVAL_MODE}')...")
inv_model = tf.keras.models.load_model(
    INVERSE_DIR / "Model_1seed" / "inverse.keras", compile=False
)
if EVAL_MODE == "forward":
    fwd_models = [
        tf.keras.models.load_model(FORWARD_DIR / f"Model_{i}seed" / "forward.keras", compile=False)
        for i in range(1, 4)
    ]
print("Modelos cargados.\n")

def make_target(f0, sigma):
    return (1.0 - A * np.exp(-0.5 * ((FREQS - f0) / sigma) ** 2)).astype(np.float32)

def predict_params(target):
    p_norm = inv_model.predict(target.reshape(1, -1), verbose=0)[0]
    return p_norm * (param_max - param_min) + param_min

def forward_nn(th1, th2, d1, d2):
    p = np.array([[th1, th2, d1, d2]], dtype=np.float32)
    p_norm = (p - fwd_param_min) / (fwd_param_max - fwd_param_min)
    return np.mean([m.predict(p_norm, verbose=0)[0] for m in fwd_models], axis=0)

def tmm_spectrum(th1, th2, d1, d2):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[MoO3(d=d1*1e-9, phi=np.deg2rad(th1)),
                MoO3(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0])
                     for f in FREQS])

def r2_window(target, pred, f0, sigma):
    mask = (FREQS >= f0 - K*sigma) & (FREQS <= f0 + K*sigma)
    if mask.sum() < 2:
        return np.nan
    t, p = target[mask], pred[mask]
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else np.nan

def mae_window(target, pred, f0, sigma):
    mask = (FREQS >= f0 - K*sigma) & (FREQS <= f0 + K*sigma)
    return float(np.mean(np.abs(target[mask] - pred[mask])))

f0_vals    = np.linspace(F0_MIN, F0_MAX, N_F0)
sigma_vals = np.linspace(SIGMA_MIN, SIGMA_MAX, N_SIGMA)

r2_grid  = np.full((N_SIGMA, N_F0), np.nan)
mae_grid = np.full((N_SIGMA, N_F0), np.nan)
results  = []

total = N_F0 * N_SIGMA
done  = 0

for j, f0 in enumerate(f0_vals):
    for i, sigma in enumerate(sigma_vals):
        if f0 - K*sigma < FREQ_MIN or f0 + K*sigma > FREQ_MAX:
            done += 1
            continue

        target           = make_target(f0, sigma)
        th1, th2, d1, d2 = predict_params(target)
        T_pred           = forward_nn(th1, th2, d1, d2)
        r2               = r2_window(target, T_pred, f0, sigma)
        mae              = mae_window(target, T_pred, f0, sigma)

        r2_grid[i, j]  = r2
        mae_grid[i, j] = mae
        results.append(dict(f0=f0, sigma=sigma, th1=th1, th2=th2, d1=d1, d2=d2,
                            target=target, T_pred=T_pred, r2=r2, mae=mae))
        done += 1
        print(f"[{done:5d}/{total}]  f0={f0:6.0f}  sigma={sigma:5.0f}  "
              f"R2={r2:.4f}  MAE={mae:.4f}")

print(f"\nCasos validos: {len(results)}")
valid_r2 = [r['r2'] for r in results if not np.isnan(r['r2'])]
print(f"R2 medio={np.mean(valid_r2):.4f}  min={np.min(valid_r2):.4f}  max={np.max(valid_r2):.4f}")

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.cm.RdYlGn.copy()
cmap.set_bad("lightgrey")
im = ax.imshow(
    r2_grid, origin="lower", aspect="auto",
    extent=[F0_MIN, F0_MAX, SIGMA_MIN, SIGMA_MAX],
    vmin=0.9, vmax=1.0, cmap=cmap,
)
plt.colorbar(im, ax=ax, label="R2 (ventana +-3sigma)")
ax.set_xlabel("f0 (cm-1)", fontsize=12)
ax.set_ylabel("sigma (cm-1)", fontsize=12)
ax.set_title(
    f"Calidad ajuste inverso -- dip gaussiano  [MoO3/MoO3/BaF2]\n"
    f"EVAL_MODE={EVAL_MODE}  |  N_train={N_TRAIN}  |  gris=fuera del espectro",
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / f"heatmap_r2_{EVAL_MODE}.png", dpi=150)
print(f"Guardado: heatmap_r2_{EVAL_MODE}.png")

results_valid  = [r for r in results if not np.isnan(r['r2'])]
results_sorted = sorted(results_valid, key=lambda r: r['r2'], reverse=True)

n_ej = min(6, len(results_sorted))
casos_buenos = results_sorted[:n_ej]
casos_malos  = results_sorted[-n_ej:]

print(f"\nCalculando TMM para {2*n_ej} casos seleccionados...")
for r in casos_buenos + casos_malos:
    r['T_tmm'] = tmm_spectrum(r['th1'], r['th2'], r['d1'], r['d2'])
    print(f"  f0={r['f0']:.0f}  sigma={r['sigma']:.0f}")

def plot_ejemplos(casos, titulo, fname):
    ncols = 3
    nrows = math.ceil(len(casos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).flat
    for k, r in enumerate(casos):
        f0, sigma = r['f0'], r['sigma']
        ax = axes[k]
        ax.plot(FREQS, r['target'], "k-",  lw=2,   label="Objetivo gaussiano")
        ax.plot(FREQS, r['T_pred'], "r--", lw=1.5, label="Forward NN")
        ax.plot(FREQS, r['T_tmm'],  "b--", lw=1.5, label="TMM")
        ax.axvspan(f0 - K*sigma, f0 + K*sigma, alpha=0.08, color="orange", label="Ventana eval")
        ax.set_xlabel("Numero de onda (cm-1)", fontsize=9)
        ax.set_ylabel("T_xx", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(
            f"f0={f0:.0f}  sigma={sigma:.0f}  "
            f"th1={r['th1']:.0f}  th2={r['th2']:.0f}  d1={r['d1']:.0f}  d2={r['d2']:.0f}\n"
            f"R2={r['r2']:.4f}  MAE={r['mae']:.4f}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for k in range(len(casos), nrows * ncols):
        axes[k].set_visible(False)
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    print(f"Guardado: {fname}")

plot_ejemplos(casos_buenos, "Mejores 6 -- dip gaussiano [MoO3/MoO3/BaF2] (eval: forward NN)", "ejemplos_buenos.png")
plot_ejemplos(casos_malos,  "Peores 6  -- dip gaussiano [MoO3/MoO3/BaF2] (eval: forward NN)", "ejemplos_malos.png")

plt.show()
print("\nFin.")