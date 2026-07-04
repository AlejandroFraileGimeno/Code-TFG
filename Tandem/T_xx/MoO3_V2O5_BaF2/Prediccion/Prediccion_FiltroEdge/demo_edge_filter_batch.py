# -*- coding: utf-8 -*-
"""
Barrido de filtros de borde -- VERSION BATCH -- T_xx  MoO3/V2O5/BaF2
======================================================================
Filtro paso-alto (HP) y paso-bajo (LP) con transicion sigmoide.

Parametros del filtro:
  f_mid   : frecuencia de corte (cm-1)
  FLANK   : anchura de la transicion (cm-1)
  T_left  : transmitancia zona izquierda
  T_right : transmitancia zona derecha

Ventana de evaluacion: [f_mid - MARGIN, f_mid + MARGIN]

Heatmap: f_mid (x) x FLANK (y), una subfigura por combo (T_left, T_right).
Salidas:
  resultados/heatmap_highpass.png
  resultados/heatmap_lowpass.png
  resultados/ejemplos_buenos_highpass.png
  resultados/ejemplos_buenos_lowpass.png
  resultados/ejemplos_malos_highpass.png
  resultados/ejemplos_malos_lowpass.png
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
    Air, BaF2, MoO3, V2O5, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
N_TRAIN = 20_000

FMID_MIN  = 550.0
FMID_MAX  = 1250.0
N_FMID    = 500

FLANK_MIN = 0.0
FLANK_MAX = 200.0
N_FLANK   = 300

MARGIN    = 400.0

FREQ_MIN  = 400.0
FREQ_MAX  = 1400.0

BATCH_SIZE = 1024

EVAL_MODE = "forward"

T_COMBOS_HP = [(0.0, 1.0), (0.025, 0.95), (0.05, 0.9)]
ALL_COMBOS  = [("highpass", Tl, Tr) for Tl, Tr in T_COMBOS_HP]
# ============================================================

FLANK_EPS = 1.0

INVERSE_DIR = ROOT_PATH / "Models" / "T_xx" / "MoO3_V2O5_BaF2" / f"Inverse_N{N_TRAIN}"
FORWARD_DIR = ROOT_PATH / "Models" / "T_xx" / "MoO3_V2O5_BaF2" / "Forward"
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

print("Cargando modelos...")
inv_model = tf.keras.models.load_model(
    INVERSE_DIR / "Model_1seed" / "inverse.keras", compile=False
)
fwd_models = [
    tf.keras.models.load_model(FORWARD_DIR / f"Model_{i}seed" / "forward.keras", compile=False)
    for i in range(1, 4)
]
print("Modelos cargados.\n")

def make_target(f_mid, flank, T_left, T_right):
    w = max(flank, FLANK_EPS)
    sig = 1.0 / (1.0 + np.exp(-(FREQS - f_mid) / (w / 6.0)))
    return np.clip(T_left + (T_right - T_left) * sig, 0, 1).astype(np.float32)

def tmm_spectrum(th1, th2, d1, d2):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[MoO3(d=d1*1e-9, phi=np.deg2rad(th1)),
                V2O5(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0]) for f in FREQS])

def eval_mask(f_mid):
    lo = max(f_mid - MARGIN, FREQ_MIN)
    hi = min(f_mid + MARGIN, FREQ_MAX)
    return (FREQS >= lo) & (FREQS <= hi)

def r2_score(target, pred, f_mid):
    mask = eval_mask(f_mid)
    if mask.sum() < 2:
        return np.nan
    t, p = target[mask], pred[mask]
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else np.nan

def mae_score(target, pred, f_mid):
    mask = eval_mask(f_mid)
    return float(np.mean(np.abs(target[mask] - pred[mask])))

fmid_vals  = np.linspace(FMID_MIN, FMID_MAX, N_FMID)
flank_vals = np.linspace(FLANK_MIN, FLANK_MAX, N_FLANK)

print("Construyendo grid de targets...")
valid_cases = []
for tc, (fname, T_left, T_right) in enumerate(ALL_COMBOS):
    for fi, f_mid in enumerate(fmid_vals):
        if f_mid - MARGIN < FREQ_MIN or f_mid + MARGIN > FREQ_MAX:
            continue
        for fj, flank in enumerate(flank_vals):
            valid_cases.append((tc, fname, T_left, T_right, fi, fj, f_mid, flank))

N_valid = len(valid_cases)
print(f"  {N_valid} casos validos")

all_targets = np.stack([make_target(f_mid, flank, T_left, T_right)
                        for (tc, fname, T_left, T_right, fi, fj, f_mid, flank) in valid_cases])

print(f"\nPrediccion inversa batch ({N_valid} muestras)...")
params_norm = inv_model.predict(all_targets, batch_size=BATCH_SIZE, verbose=1)
all_params  = params_norm * (param_max - param_min) + param_min

print("\nPrediccion forward batch...")
params_fwd_norm = ((all_params - fwd_param_min) /
                   (fwd_param_max - fwd_param_min)).astype(np.float32)
all_T_pred = np.mean(
    [m.predict(params_fwd_norm, batch_size=BATCH_SIZE, verbose=1) for m in fwd_models], axis=0
)

print("\nCalculando metricas...")
r2_grid = np.full((len(ALL_COMBOS), N_FMID, N_FLANK), np.nan)
all_results = []

for k, (tc, fname, T_left, T_right, fi, fj, f_mid, flank) in enumerate(valid_cases):
    target           = all_targets[k]
    T_pred           = all_T_pred[k]
    th1, th2, d1, d2 = all_params[k]
    r2  = r2_score(target, T_pred, f_mid)
    mae = mae_score(target, T_pred, f_mid)
    r2_grid[tc, fi, fj] = r2
    all_results.append(dict(
        filter=fname, T_left=T_left, T_right=T_right,
        f_mid=f_mid, flank=flank,
        th1=th1, th2=th2, d1=d1, d2=d2,
        target=target, T_pred=T_pred, r2=r2, mae=mae,
    ))

valid_r2 = [r['r2'] for r in all_results if not np.isnan(r['r2'])]
print(f"\nTotal validos: {len(valid_r2)}")
print(f"R2 medio={np.mean(valid_r2):.4f}  min={np.min(valid_r2):.4f}  max={np.max(valid_r2):.4f}")

cmap = plt.cm.viridis.copy()
cmap.set_bad("#e1e0d9")

for filter_name, combos, tc_offset in [
    ("highpass", T_COMBOS_HP, 0),
]:
    ncols = len(combos)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ci, (T_left, T_right) in enumerate(combos):
        tc = tc_offset + ci
        ax = axes[ci]
        grid = r2_grid[tc, :, :].T
        im = ax.imshow(
            grid, origin="lower", aspect="auto",
            extent=[FMID_MIN, FMID_MAX, FLANK_MIN, FLANK_MAX],
            vmin=0.5, vmax=1.0, cmap=cmap,
        )
        ax.set_title(f"T_left={T_left:.1f}  T_right={T_right:.1f}", fontsize=10)
        ax.set_xlabel("f_mid (cm$^{-1}$)", fontsize=9)
        if ci == 0:
            ax.set_ylabel("FLANK (cm$^{-1}$)", fontsize=9)
    fig.colorbar(im, ax=axes[-1], label=f"R2 (ventana +-{MARGIN:.0f} cm-1)")
    label = "Paso-alto (high-pass)" if filter_name == "highpass" else "Paso-bajo (low-pass)"
    fig.suptitle(f"Filtro de borde -- {label}  [MoO3/V2O5/BaF2]  EVAL={EVAL_MODE}", fontsize=12)
    fig.tight_layout()
    fname_out = f"heatmap_{filter_name}.png"
    fig.savefig(OUT_DIR / fname_out, dpi=200, bbox_inches="tight")
    print(f"Guardado: {fname_out}")

def plot_ejemplos(casos, titulo, fname):
    ncols = 3
    nrows = math.ceil(len(casos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).flat
    for k, r in enumerate(casos):
        f_mid = r['f_mid']
        ax = axes[k]
        ax.plot(FREQS, r['target'], color="#0b0b0b", lw=2.0, label="Objetivo")
        ax.plot(FREQS, r['T_pred'], color="#e34948", lw=1.4, ls="--", label="Forward NN")
        ax.plot(FREQS, r['T_tmm'],  color="#2a78d6", lw=1.6, label="TMM")
        lo = max(f_mid - MARGIN, FREQ_MIN)
        hi = min(f_mid + MARGIN, FREQ_MAX)
        ax.axvspan(lo, hi, alpha=0.5, color="#e1e0d9", label="Ventana eval")
        ax.axvline(f_mid, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=9)
        ax.set_ylabel(r"$T_{xx}$", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(
            f"f_mid={f_mid:.0f}  FLANK={r['flank']:.0f}  "
            f"Tl={r['T_left']:.1f}  Tr={r['T_right']:.1f}\n"
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

for filter_name in ("highpass",):
    subset = [r for r in all_results if r['filter'] == filter_name and not np.isnan(r['r2'])]
    sorted_sub = sorted(subset, key=lambda r: r['r2'], reverse=True)
    n_ej = min(6, len(sorted_sub))
    buenos = sorted_sub[:n_ej]
    malos  = sorted_sub[-n_ej:]
    print(f"\nCalculando TMM para {filter_name} ({2*n_ej} casos)...")
    for r in buenos + malos:
        r['T_tmm'] = tmm_spectrum(r['th1'], r['th2'], r['d1'], r['d2'])
    label = "paso-alto" if filter_name == "highpass" else "paso-bajo"
    plot_ejemplos(buenos, f"Mejores 6 -- filtro {label} [MoO3/V2O5/BaF2]", f"ejemplos_buenos_{filter_name}.png")
    plot_ejemplos(malos,  f"Peores 6  -- filtro {label} [MoO3/V2O5/BaF2]", f"ejemplos_malos_{filter_name}.png")

plt.show()
print("\nFin.")