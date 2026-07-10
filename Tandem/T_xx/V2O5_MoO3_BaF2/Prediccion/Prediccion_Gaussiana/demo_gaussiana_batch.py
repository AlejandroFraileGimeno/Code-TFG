# -*- coding: utf-8 -*-
"""
Barrido de dips gaussianos -- VERSION BATCH -- diseno inverso T_xx  V2O5/MoO3/BaF2
====================================================================================
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

# ---------------------------------------------------------------------------
# Estilo TFG (solo estética; no afecta a los cálculos)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           15,
    "axes.labelsize":      18,
    "axes.titlesize":      15,
    "xtick.labelsize":     14,
    "ytick.labelsize":     14,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     14,
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

BATCH_SIZE = 1024

EVAL_MODE  = "forward"
# ============================================================

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

def tmm_spectrum(th1, th2, d1, d2):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[V2O5(d=d1*1e-9, phi=np.deg2rad(th1)),
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

print("Construyendo grid de targets...")
valid_idx = []
valid_f0  = []
valid_sig = []

for j, f0 in enumerate(f0_vals):
    for i, sigma in enumerate(sigma_vals):
        if f0 - K*sigma < FREQ_MIN or f0 + K*sigma > FREQ_MAX:
            continue
        valid_idx.append((i, j))
        valid_f0.append(f0)
        valid_sig.append(sigma)

N_valid = len(valid_idx)
print(f"  {N_valid} casos validos de {N_F0 * N_SIGMA} totales")

all_targets = np.stack([make_target(f0, sig) for f0, sig in zip(valid_f0, valid_sig)])

print(f"\nPrediccion inversa batch ({N_valid} muestras)...")
params_norm = inv_model.predict(all_targets, batch_size=BATCH_SIZE, verbose=1)
all_params  = params_norm * (param_max - param_min) + param_min

if EVAL_MODE == "forward":
    print("\nPrediccion forward batch...")
    params_fwd_norm = ((all_params - fwd_param_min) /
                       (fwd_param_max - fwd_param_min)).astype(np.float32)
    all_T_pred = np.mean(
        [m.predict(params_fwd_norm, batch_size=BATCH_SIZE, verbose=1) for m in fwd_models],
        axis=0
    )

print("\nCalculando metricas...")
r2_grid  = np.full((N_SIGMA, N_F0), np.nan)
mae_grid = np.full((N_SIGMA, N_F0), np.nan)
results  = []

for k, ((i, j), f0, sigma) in enumerate(zip(valid_idx, valid_f0, valid_sig)):
    target           = all_targets[k]
    T_pred           = all_T_pred[k]
    th1, th2, d1, d2 = all_params[k]
    r2  = r2_window(target, T_pred, f0, sigma)
    mae = mae_window(target, T_pred, f0, sigma)
    r2_grid[i, j]  = r2
    mae_grid[i, j] = mae
    results.append(dict(f0=f0, sigma=sigma, th1=th1, th2=th2, d1=d1, d2=d2,
                        target=target, T_pred=T_pred, r2=r2, mae=mae))

print(f"\nCasos validos: {len(results)}")
valid_r2 = [r['r2'] for r in results if not np.isnan(r['r2'])]
print(f"R2 medio={np.mean(valid_r2):.4f}  min={np.min(valid_r2):.4f}  max={np.max(valid_r2):.4f}")

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.cm.viridis.copy()
cmap.set_bad("#e1e0d9")
im = ax.imshow(
    r2_grid, origin="lower", aspect="auto",
    extent=[F0_MIN, F0_MAX, SIGMA_MIN, SIGMA_MAX],
    vmin=0.0, vmax=1.0, cmap=cmap,
)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$R^2$ (ventana $\pm3\sigma$)", fontsize=18)
ax.set_xlabel(r"$\omega_0$ (cm$^{-1}$)")
ax.set_ylabel(r"$\sigma$ (cm$^{-1}$)")
fig.tight_layout()
fig.savefig(OUT_DIR / f"heatmap_r2_{EVAL_MODE}.png", dpi=200, bbox_inches="tight")
ARREGLOS = ROOT_PATH / "Arreglos en Gráficos"
ARREGLOS.mkdir(exist_ok=True)
fig.savefig(ARREGLOS / "heatmap_r2_gaussiana_V2O5_MoO3.png", dpi=200, bbox_inches="tight")
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
        ax.plot(FREQS, r['target'], color="#0b0b0b", lw=2.0, label="Objetivo gaussiano")
        ax.plot(FREQS, r['T_pred'], color="#e34948", lw=1.4, ls="--", label="Surrogate")
        ax.plot(FREQS, r['T_tmm'],  color="#2a78d6", lw=1.6, label="Simulación")
        ax.axvspan(f0 - K*sigma, f0 + K*sigma, alpha=0.5, color="#e1e0d9", label="Ventana eval")
        ax.set_xlabel(r"$\omega$ (cm$^{-1}$)", fontsize=9)
        ax.set_ylabel(r"$T_{xx}$", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(
            rf"$\omega_0={f0:.0f}$ cm$^{{-1}}$, $\sigma={sigma:.0f}$ cm$^{{-1}}$" "\n"
            rf"$\phi_1={r['th1']:.0f}^\circ$, $\phi_2={r['th2']:.0f}^\circ$, "
            rf"$d_1={r['d1']:.0f}$ nm, $d_2={r['d2']:.0f}$ nm" "\n"
            rf"$R^2={r['r2']:.3f}$", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True)
    for k in range(len(casos), nrows * ncols):
        axes[k].set_visible(False)
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=200, bbox_inches="tight")
    print(f"Guardado: {fname}")

# ---------------------------------------------------------------------------
# Figura individual del mejor caso (estilo TFG)
# ---------------------------------------------------------------------------
mejor = casos_buenos[0]
figm, axm = plt.subplots(figsize=(5.5, 5.5))
axm.axvspan(mejor['f0'] - K*mejor['sigma'], mejor['f0'] + K*mejor['sigma'],
            alpha=0.5, color="#e1e0d9", zorder=0, label="Ventana eval")
axm.plot(FREQS, mejor['target'], color="#0b0b0b", lw=2.0, label="Objetivo")
axm.plot(FREQS, mejor['T_tmm'],  color="#2a78d6", lw=1.6, label="Simulación")
axm.plot(FREQS, mejor['T_pred'], color="#e34948", lw=1.4, ls="--", label="Surrogate")
axm.set_title(
    "V$_2$O$_5$ / MoO$_3$\n"
    + rf"$\omega_0={mejor['f0']:.0f}$ cm$^{{-1}}$, $\sigma={mejor['sigma']:.0f}$ cm$^{{-1}}$, $R^2={mejor['r2']:.3f}$" + "\n"
    + rf"$\phi_1={mejor['th1']:.0f}^\circ$, $\phi_2={mejor['th2']:.0f}^\circ$, $d_1={mejor['d1']:.0f}$ nm, $d_2={mejor['d2']:.0f}$ nm",
    pad=8, fontsize=13,
)
axm.set_xlabel(r"$\omega$ (cm$^{-1}$)")
axm.set_ylabel(r"$T_{xx}$")
axm.set_xlim(FREQS[0], FREQS[-1])
axm.set_ylim(-0.02, 1.05)
axm.legend(loc="lower right")
figm.tight_layout()
figm.savefig(ARREGLOS / "mejor_caso_gaussiana_V2O5_MoO3.png", dpi=200, bbox_inches="tight")
print("Guardado: mejor_caso_gaussiana_V2O5_MoO3.png")

plot_ejemplos(casos_buenos, "Mejores 6 -- dip gaussiano [V2O5/MoO3/BaF2] (eval: forward NN)", "ejemplos_buenos.png")
plot_ejemplos(casos_malos,  "Peores 6  -- dip gaussiano [V2O5/MoO3/BaF2] (eval: forward NN)", "ejemplos_malos.png")

plt.show()
print("\nFin.")