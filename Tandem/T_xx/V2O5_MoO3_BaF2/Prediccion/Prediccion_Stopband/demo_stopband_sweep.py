# -*- coding: utf-8 -*-
"""
Barrido de stop-bands — diseno inverso T_xx  V2O5/MoO3/BaF2
============================================================
Para cada combinacion (left, right, A_pct, flank) construye el objetivo:

    T ~ 1        fuera de la banda
    T ~ 1-A_pct  dentro de la banda   (A_pct en %)

Los flancos se suavizan con una sigmoide de anchura FLANK (cm-1).
Si FLANK=0 se usa un escalon ideal.

Ejes barridos:
  left   : borde izquierdo (cm-1)
  right  : borde derecho   (cm-1)   [right > left + MIN_WIDTH]
  A_pct  : porcentaje de supresion  [50,60,70,80,90,100] %
  flank  : anchura del flanco       [0,25,50,75,100] cm-1

Salidas por cada valor de A_pct:
  resultados/heatmap_A<X>_<EVAL_MODE>.png   -- subfigura por flank
  resultados/ejemplos_A<X>_<EVAL_MODE>.png  -- mejores 6 casos (todos los flanks)

EVAL_MODE = "forward"  ->  forward NN  (rapido, metrica heatmap)
          = "tmm"      ->  TMM exacto  (lento)

Los paneles de ejemplos muestran siempre TMM + forward NN.
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
    Air, BaF2, MoO3, V2O5, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
N_TRAIN   = 20_000

A_PCT_LIST   = [50, 60, 70, 80, 90, 100]    # % de transmitancia fuera de la banda
FLANK_LIST   = [0, 25, 50, 75, 100]         # cm-1

MARGIN    = 100.0   # margen fuera de la banda para evaluar (cm-1)

LEFT_MIN  = 500.0
LEFT_MAX  = 1100.0
N_LEFT    = 25

RIGHT_MIN = 550.0
RIGHT_MAX = 1300.0
N_RIGHT   = 25

MIN_WIDTH = 50.0    # anchura minima de la banda (cm-1)

FREQ_MIN  = 400.0
FREQ_MAX  = 1400.0

EVAL_MODE = "forward"   # "forward" o "tmm"
# ============================================================

FLANK_EPS = 1.0   # flanco minimo interno (evita division por 0)

INVERSE_DIR = ROOT_PATH / "Models" / "T_xx" / "V2O5_MoO3_BaF2" / f"Inverse_N{N_TRAIN}"
FORWARD_DIR = ROOT_PATH / "Models" / "T_xx" / "V2O5_MoO3_BaF2" / "Forward"
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

print(f"Cargando modelos (EVAL_MODE='{EVAL_MODE}')...")
inv_model = tf.keras.models.load_model(
    INVERSE_DIR / "Model_1seed" / "inverse.keras", compile=False
)
fwd_models = [
    tf.keras.models.load_model(FORWARD_DIR / f"Model_{i}seed" / "forward.keras", compile=False)
    for i in range(1, 4)
]
print("Modelos cargados.\n")

# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------
def _sig(f, center, width):
    w = max(width, FLANK_EPS)
    return 1.0 / (1.0 + np.exp((f - center) / (w / 6.0)))

def make_target(left, right, a_pct, flank):
    # T_outside = A = a_pct/100,  T_inside = 0
    T_outside = a_pct / 100.0
    w = flank if flank > 0 else FLANK_EPS
    down   = 1.0 - _sig(FREQS, left,  w)   # ~1 despues de left
    up     = _sig(FREQS, right, w)           # ~1 antes de right
    inside = down * up                       # ~1 dentro de la banda
    T = T_outside * (1.0 - inside)          # A fuera, 0 dentro
    return np.clip(T, 0, 1).astype(np.float32)

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
        layers=[V2O5(d=d1*1e-9, phi=np.deg2rad(th1)),
                MoO3(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0])
                     for f in FREQS])

def evaluate(th1, th2, d1, d2):
    return forward_nn(th1, th2, d1, d2) if EVAL_MODE == "forward" else tmm_spectrum(th1, th2, d1, d2)

def eval_mask(left, right):
    lo = max(left  - MARGIN, FREQ_MIN)
    hi = min(right + MARGIN, FREQ_MAX)
    return (FREQS >= lo) & (FREQS <= hi)

def r2_window(target, pred, left, right):
    mask = eval_mask(left, right)
    if mask.sum() < 2:
        return np.nan
    t, p = target[mask], pred[mask]
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else np.nan

def mae_window(target, pred, left, right):
    mask = eval_mask(left, right)
    return float(np.mean(np.abs(target[mask] - pred[mask])))

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
left_vals  = np.linspace(LEFT_MIN,  LEFT_MAX,  N_LEFT)
right_vals = np.linspace(RIGHT_MIN, RIGHT_MAX, N_RIGHT)

# r2_grid[i_A, i_flank, i_left, i_right]
r2_grid = np.full((len(A_PCT_LIST), len(FLANK_LIST), N_LEFT, N_RIGHT), np.nan)

# todos los resultados
all_results = []   # lista de dicts

total_combos = len(A_PCT_LIST) * len(FLANK_LIST) * N_LEFT * N_RIGHT
done = 0

for ka, a_pct in enumerate(A_PCT_LIST):
    for kf, flank in enumerate(FLANK_LIST):
        for i, left in enumerate(left_vals):
            for j, right in enumerate(right_vals):
                done += 1
                if right <= left + MIN_WIDTH:
                    continue
                if left - MARGIN < FREQ_MIN or right + MARGIN > FREQ_MAX:
                    continue

                target           = make_target(left, right, a_pct, flank)
                th1, th2, d1, d2 = predict_params(target)
                T_pred           = forward_nn(th1, th2, d1, d2)
                r2               = r2_window(target, T_pred, left, right)
                mae              = mae_window(target, T_pred, left, right)

                r2_grid[ka, kf, i, j] = r2
                all_results.append(dict(
                    a_pct=a_pct, flank=flank, left=left, right=right,
                    th1=th1, th2=th2, d1=d1, d2=d2,
                    target=target, T_pred=T_pred,
                    r2=r2, mae=mae,
                ))
                if done % 50 == 0 or done == total_combos:
                    print(f"[{done:5d}/{total_combos}]  A={a_pct}%  flank={flank}  "
                          f"left={left:.0f}  right={right:.0f}  R2={r2:.3f}")

print(f"\nCasos validos: {len(all_results)}")
valid_r2 = [r['r2'] for r in all_results if not np.isnan(r['r2'])]
if valid_r2:
    print(f"R2 medio={np.mean(valid_r2):.4f}  min={np.min(valid_r2):.4f}  max={np.max(valid_r2):.4f}")

# ---------------------------------------------------------------------------
# Heatmaps: una figura por A, un subplot por flank
# ---------------------------------------------------------------------------
for ka, a_pct in enumerate(A_PCT_LIST):
    ncols = len(FLANK_LIST)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 5), sharey=True)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgrey")
    for kf, flank in enumerate(FLANK_LIST):
        ax  = axes[kf]
        grid = r2_grid[ka, kf, :, :]
        im = ax.imshow(
            grid, origin="lower", aspect="auto",
            extent=[RIGHT_MIN, RIGHT_MAX, LEFT_MIN, LEFT_MAX],
            vmin=0.5, vmax=1.0, cmap=cmap,
        )
        ax.set_title(f"flank={flank} cm-1", fontsize=10)
        ax.set_xlabel("right (cm-1)", fontsize=9)
        if kf == 0:
            ax.set_ylabel("left (cm-1)", fontsize=9)
        ax.plot([RIGHT_MIN, RIGHT_MAX],
                [RIGHT_MIN - MIN_WIDTH, RIGHT_MAX - MIN_WIDTH],
                "k--", lw=0.8, alpha=0.4)
    fig.colorbar(im, ax=axes[-1], label=f"R2 (+-{MARGIN:.0f} cm-1)")
    fig.suptitle(
        f"Stop-band sweep — T_outside={a_pct}%  T_inside=0  [V2O5/MoO3/BaF2]  EVAL={EVAL_MODE}",
        fontsize=12)
    fig.tight_layout()
    fname = f"heatmap_A{a_pct}_{EVAL_MODE}.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    print(f"Guardado: {fname}")

# ---------------------------------------------------------------------------
# Ejemplos: mejores 6 de todos los casos
# ---------------------------------------------------------------------------
results_sorted = sorted(all_results, key=lambda r: r['r2'] if not np.isnan(r['r2']) else -999, reverse=True)
label_pred = "Forward NN" if EVAL_MODE == "forward" else "TMM"

def plot_ejemplos(casos, titulo, fname):
    ncols = 3
    nrows = math.ceil(len(casos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).flat
    for k, r in enumerate(casos):
        left, right = r['left'], r['right']
        ax = axes[k]
        ax.plot(FREQS, r['target'], "k-",  lw=2,   label="Objetivo stop-band")
        ax.plot(FREQS, r['T_pred'], "r--", lw=1.5, label="Forward NN")
        ax.plot(FREQS, r['T_tmm'],  "b--", lw=1.5, label="TMM")
        lo = max(left - MARGIN, FREQ_MIN)
        hi = min(right + MARGIN, FREQ_MAX)
        ax.axvspan(lo, hi, alpha=0.06, color="orange", label="Ventana eval")
        ax.axvspan(left, right, alpha=0.08, color="red", label="Banda")
        ax.set_xlabel("Numero de onda (cm-1)", fontsize=9)
        ax.set_ylabel("T_xx", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(
            f"left={left:.0f}  right={right:.0f}  A={r['a_pct']}%  flank={r['flank']}\n"
            f"th1={r['th1']:.0f}  th2={r['th2']:.0f}  d1={r['d1']:.0f}  d2={r['d2']:.0f}\n"
            f"R2={r['r2']:.4f}  MAE={r['mae']:.4f}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for k in range(len(casos), nrows * ncols):
        axes[k].set_visible(False)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    print(f"Guardado: {fname}")

n_ej = min(6, len(results_sorted))
casos_buenos = results_sorted[:n_ej]
casos_malos  = results_sorted[-n_ej:]

print(f"\nCalculando TMM para {2*n_ej} casos seleccionados...")
for r in casos_buenos + casos_malos:
    r['T_tmm'] = tmm_spectrum(r['th1'], r['th2'], r['d1'], r['d2'])
    print(f"  left={r['left']:.0f}  right={r['right']:.0f}  A={r['a_pct']}%  flank={r['flank']}")

plot_ejemplos(casos_buenos,
              "Mejores 6 stop-bands [V2O5/MoO3/BaF2] (eval: forward NN)",
              "ejemplos_buenos.png")
plot_ejemplos(casos_malos,
              "Peores 6 stop-bands [V2O5/MoO3/BaF2] (eval: forward NN)",
              "ejemplos_malos.png")

plt.show()
print("\nFin.")