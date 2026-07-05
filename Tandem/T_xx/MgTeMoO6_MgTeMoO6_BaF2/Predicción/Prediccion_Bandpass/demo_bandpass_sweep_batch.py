# -*- coding: utf-8 -*-
"""
Barrido de pasa-bandas -- VERSION BATCH -- diseno inverso T_xx  MgTeMoO6/MgTeMoO6/BaF2
=========================================================================================
Salidas:
  resultados/heatmap_Tlow<X>_<EVAL_MODE>.png
  resultados/ejemplos_buenos.png
  resultados/ejemplos_malos.png
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Estilo TFG (solo estética; no afecta a los cálculos)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           12,
    "axes.labelsize":      13,
    "axes.titlesize":      12,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     11,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MgTeMoO6, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
N_TRAIN = 20_000

T_LOW_LIST       = [0.0, 0.10, 0.20]
OUTER_WIDTH_LIST = [10, 20, 50]
T_HIGH           = 1.0
FLANK            = 50.0

MARGIN    = 100.0

LE_MIN  = 600.0
LE_MAX  = 800.0
N_LE    = 50

RS_MIN  = 800.0
RS_MAX  = 1000.0
N_RS    = 50

MIN_PASS_WIDTH = 50.0

FREQ_MIN = 400.0
FREQ_MAX = 1400.0

BATCH_SIZE = 1024

EVAL_MODE = "forward"
# ============================================================

FLANK_EPS = 1.0

INVERSE_DIR = ROOT_PATH / "Models" / "T_xx" / "MgTeMoO6_MgTeMoO6_BaF2" / f"Inverse_N{N_TRAIN}"
FORWARD_DIR = ROOT_PATH / "Models" / "T_xx" / "MgTeMoO6_MgTeMoO6_BaF2" / "Forward"
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

def _sig(freqs, center, width):
    w = max(width, FLANK_EPS)
    return 1.0 / (1.0 + np.exp(-(freqs - center) / (w / 6.0)))

def make_target(le, rs, T_low, T_high, outer_width, flank):
    w = max(flank, FLANK_EPS)
    rise = _sig(FREQS, le, w)
    fall = _sig(FREQS, rs, w)
    passband = rise * (1.0 - fall)
    T = T_low + (T_high - T_low) * passband
    return np.clip(T, 0, 1).astype(np.float32)

def tmm_spectrum(th1, th2, d1, d2):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[MgTeMoO6(d=d1*1e-9, phi=np.deg2rad(th1)),
                MgTeMoO6(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0])
                     for f in FREQS])

def eval_mask(le, rs, outer_width):
    ls = le - outer_width
    re = rs + outer_width
    lo = max(ls - MARGIN, FREQ_MIN)
    hi = min(re + MARGIN, FREQ_MAX)
    return (FREQS >= lo) & (FREQS <= hi)

def r2_window(target, pred, le, rs, outer_width):
    mask = eval_mask(le, rs, outer_width)
    if mask.sum() < 2:
        return np.nan
    t, p = target[mask], pred[mask]
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else np.nan

def mae_window(target, pred, le, rs, outer_width):
    mask = eval_mask(le, rs, outer_width)
    return float(np.mean(np.abs(target[mask] - pred[mask])))

le_vals = np.linspace(LE_MIN, LE_MAX, N_LE)
rs_vals = np.linspace(RS_MIN, RS_MAX, N_RS)

print("Construyendo grid de targets...")
valid_cases = []
for kt, T_low in enumerate(T_LOW_LIST):
    for kw, outer_width in enumerate(OUTER_WIDTH_LIST):
        for i, le in enumerate(le_vals):
            for j, rs in enumerate(rs_vals):
                ls = le - outer_width
                re = rs + outer_width
                if rs <= le + MIN_PASS_WIDTH:
                    continue
                if ls < FREQ_MIN + MARGIN or re > FREQ_MAX - MARGIN:
                    continue
                valid_cases.append((kt, kw, i, j, T_low, outer_width, le, rs))

N_valid = len(valid_cases)
print(f"  {N_valid} casos validos de {len(T_LOW_LIST)*len(OUTER_WIDTH_LIST)*N_LE*N_RS} totales")

all_targets = np.stack([make_target(le, rs, T_low, T_HIGH, outer_width, FLANK)
                        for (kt, kw, i, j, T_low, outer_width, le, rs) in valid_cases])

print(f"\nPrediccion inversa batch ({N_valid} muestras)...")
params_norm = inv_model.predict(all_targets, batch_size=BATCH_SIZE, verbose=1)
all_params  = params_norm * (param_max - param_min) + param_min

print("\nPrediccion forward batch...")
params_fwd_norm = ((all_params - fwd_param_min) /
                   (fwd_param_max - fwd_param_min)).astype(np.float32)
all_T_pred = np.mean(
    [m.predict(params_fwd_norm, batch_size=BATCH_SIZE, verbose=1) for m in fwd_models],
    axis=0
)

print("\nCalculando metricas...")
r2_grid  = np.full((len(T_LOW_LIST), len(OUTER_WIDTH_LIST), N_LE, N_RS), np.nan)
all_results = []

for k, (kt, kw, i, j, T_low, outer_width, le, rs) in enumerate(valid_cases):
    target           = all_targets[k]
    T_pred           = all_T_pred[k]
    th1, th2, d1, d2 = all_params[k]
    ls = le - outer_width
    re = rs + outer_width
    r2  = r2_window(target, T_pred, le, rs, outer_width)
    mae = mae_window(target, T_pred, le, rs, outer_width)
    r2_grid[kt, kw, i, j] = r2
    all_results.append(dict(
        T_low=T_low, outer_width=outer_width,
        le=le, rs=rs, ls=ls, re=re,
        th1=th1, th2=th2, d1=d1, d2=d2,
        target=target, T_pred=T_pred, r2=r2, mae=mae,
    ))

print(f"\nCasos validos: {len(all_results)}")
valid_r2 = [r['r2'] for r in all_results if not np.isnan(r['r2'])]
if valid_r2:
    print(f"R2 medio={np.mean(valid_r2):.4f}  min={np.min(valid_r2):.4f}  max={np.max(valid_r2):.4f}")

for kt, T_low in enumerate(T_LOW_LIST):
    ncols = len(OUTER_WIDTH_LIST)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 5), sharey=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e1e0d9")
    for kw, ow in enumerate(OUTER_WIDTH_LIST):
        ax   = axes[kw]
        grid = r2_grid[kt, kw, :, :]
        im = ax.imshow(
            grid, origin="lower", aspect="auto",
            extent=[RS_MIN, RS_MAX, LE_MIN, LE_MAX],
            vmin=0.5, vmax=1.0, cmap=cmap,
        )
        ax.set_title(f"zona_baja={ow:.0f} cm-1", fontsize=10)
        ax.set_xlabel("rs — borde der. pasabanda (cm$^{-1}$)", fontsize=9)
        if kw == 0:
            ax.set_ylabel("le — borde izq. pasabanda (cm$^{-1}$)", fontsize=9)
        ax.plot([RS_MIN, RS_MAX],
                [RS_MIN - MIN_PASS_WIDTH, RS_MAX - MIN_PASS_WIDTH],
                "k--", lw=0.8, alpha=0.4)
    fig.colorbar(im, ax=axes[-1], label=f"R2 (+-{MARGIN:.0f} cm-1 extra)")
    fig.suptitle(
        f"Pasa-banda -- T_low={T_low:.2f}  T_high={T_HIGH:.1f}  flank={FLANK:.0f}  "
        f"[MgTeMoO6/MgTeMoO6/BaF2]  EVAL={EVAL_MODE}",
        fontsize=11)
    fig.tight_layout()
    fname = f"heatmap_Tlow{int(T_low*100):02d}_{EVAL_MODE}.png"
    fig.savefig(OUT_DIR / fname, dpi=200, bbox_inches="tight")
    print(f"Guardado: {fname}")

results_sorted = sorted(all_results, key=lambda r: r['r2'] if not np.isnan(r['r2']) else -999, reverse=True)

def plot_ejemplos(casos, titulo, fname):
    ncols = 3
    nrows = math.ceil(len(casos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).flat
    for k, r in enumerate(casos):
        le, rs, ls, re = r['le'], r['rs'], r['ls'], r['re']
        ax = axes[k]
        ax.plot(FREQS, r['target'], color="#0b0b0b", lw=2.0, label="Objetivo pasa-banda")
        ax.plot(FREQS, r['T_pred'], color="#e34948", lw=1.4, ls="--", label="Forward NN")
        ax.plot(FREQS, r['T_tmm'],  color="#2a78d6", lw=1.6, label="TMM")
        lo = max(ls - MARGIN, FREQ_MIN)
        hi = min(re + MARGIN, FREQ_MAX)
        ax.axvspan(lo, hi, alpha=0.5, color="#e1e0d9", label="Ventana eval")
        ax.axvspan(le, rs, alpha=0.12, color="#1baf7a", label="Banda de paso")
        ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=9)
        ax.set_ylabel(r"$T_{xx}$", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(
            f"le={le:.0f}  rs={rs:.0f}  OW={r['outer_width']:.0f}  T_low={r['T_low']:.2f}\n"
            f"th1={r['th1']:.0f}  th2={r['th2']:.0f}  d1={r['d1']:.0f}  d2={r['d2']:.0f}\n"
            f"R2={r['r2']:.4f}  MAE={r['mae']:.4f}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True)
    for k in range(len(casos), nrows * ncols):
        axes[k].set_visible(False)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=200, bbox_inches="tight")
    print(f"Guardado: {fname}")

n_ej = min(6, len(results_sorted))
casos_buenos = results_sorted[:n_ej]
casos_malos  = results_sorted[-n_ej:]

print(f"\nCalculando TMM para {2*n_ej} casos seleccionados...")
for r in casos_buenos + casos_malos:
    r['T_tmm'] = tmm_spectrum(r['th1'], r['th2'], r['d1'], r['d2'])
    print(f"  le={r['le']:.0f}  rs={r['rs']:.0f}  OW={r['outer_width']:.0f}  T_low={r['T_low']:.2f}")

plot_ejemplos(casos_buenos,
              "Mejores 6 pasa-bandas [MgTeMoO6/MgTeMoO6/BaF2] (eval: forward NN)",
              "ejemplos_buenos.png")
plot_ejemplos(casos_malos,
              "Peores 6 pasa-bandas [MgTeMoO6/MgTeMoO6/BaF2] (eval: forward NN)",
              "ejemplos_malos.png")

plt.show()
print("\nFin.")