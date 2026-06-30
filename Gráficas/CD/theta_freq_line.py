# -*- coding: utf-8 -*-
"""
Extrae el CD y la Reflectancia a lo largo de la cresta del water-plot.

Para cada phi en [PHI_MIN, PHI_MAX], busca la frecuencia de maximo CD
(cresta real). Muestra:
  - Water-plot con la cresta superpuesta + ajuste lineal
  - CD a lo largo de la cresta vs phi / frecuencia
  - Reflectancia a lo largo de la cresta vs phi / frecuencia
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — edita aquí antes de ejecutar
# ============================================================
PAIR      = "MoO3_V2O5"
D1_NM     = 400
D2_NM     = 900

# Rango de phi donde buscar la cresta (grados)
PHI_MIN = 35
PHI_MAX = 85

THETA_STEP = 1        # paso del water-plot (grados)
USE_TMM    = True     # True = TMM | False = NN
NUM_SEEDS  = 1
# ============================================================

ROOT_PATH = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT_PATH / "TMM"))
from generalized_transfer_matrix_method import (
    Air, Au, MgTeMoO6, MoO3, V2O5, LayeredStructure,
    calculate_circular_dichroism_ref,
)

_SURROGATE_MAP = {
    "MgTeMoO6_MgTeMoO6": "MgTeMoO6, MgTeMoO6",
    "MoO3_MgTeMoO6":     "MoO3, MgTeMoO6",
    "MoO3_MoO3":         "MoO3, MoO3",
    "MoO3_V2O5":         "MoO3, V2O5",
    "V2O5_V2O5":         "V2O5, V2O5",
    "V2O5_MgTeMoO6":     "V2O5, MgTeMoO6",
    "MgTeMoO6_MoO3":     "MgTeMoO6, MoO3",
    "MgTeMoO6_V2O5":     "MgTeMoO6, V2O5",
    "V2O5_MoO3":         "V2O5, MoO3",
}

def _build_layers(pair, d1_m, d2_m, phi_rad):
    L = {
        "MgTeMoO6_MgTeMoO6": (MgTeMoO6(d=d1_m, phi=phi_rad), MgTeMoO6(d=d2_m)),
        "MoO3_MgTeMoO6":     (MoO3(d=d1_m,     phi=phi_rad), MgTeMoO6(d=d2_m)),
        "MoO3_MoO3":         (MoO3(d=d1_m,     phi=phi_rad), MoO3(d=d2_m)),
        "MoO3_V2O5":         (MoO3(d=d1_m,     phi=phi_rad), V2O5(d=d2_m)),
        "V2O5_V2O5":         (V2O5(d=d1_m,     phi=phi_rad), V2O5(d=d2_m)),
        "V2O5_MgTeMoO6":     (V2O5(d=d1_m,     phi=phi_rad), MgTeMoO6(d=d2_m)),
        "MgTeMoO6_MoO3":     (MgTeMoO6(d=d1_m, phi=phi_rad), MoO3(d=d2_m)),
        "MgTeMoO6_V2O5":     (MgTeMoO6(d=d1_m, phi=phi_rad), V2O5(d=d2_m)),
        "V2O5_MoO3":         (V2O5(d=d1_m,     phi=phi_rad), MoO3(d=d2_m)),
    }
    return L[pair]

# Rango de frecuencias
_CD_MODEL_DIR = ROOT_PATH / "Models" / "CD" / PAIR
_RT_MODEL_DIR = ROOT_PATH / "Models" / "R_total" / PAIR
_scaler_ref   = json.load(open(_CD_MODEL_DIR / "Model_1seed" / "scalers.json"))
FREQ_MIN = _scaler_ref["feature_min"][3]
FREQ_MAX = _scaler_ref["feature_max"][3]
N_FREQS  = _scaler_ref["n_freqs"]
FREQS    = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
thetas_all = np.arange(0, 180 + THETA_STEP, THETA_STEP)

# ---------------------------------------------------------------------------
# Cargar modelos NN si hace falta
# ---------------------------------------------------------------------------
if not USE_TMM:
    sys.path.insert(0, str(ROOT_PATH / "Surrogates" / "CD" / _SURROGATE_MAP[PAIR]))
    import utils_nn_forward as auxf
    from tensorflow.keras import models as tf_models
    _DATABASE_CD = str(ROOT_PATH / "Datasets" / "CD"      / PAIR)
    _DATABASE_RT = str(ROOT_PATH / "Datasets" / "R_total" / PAIR)
    models_cd, scalers_cd = [], []
    models_rt, scalers_rt = [], []
    for i in range(1, NUM_SEEDS + 1):
        models_cd.append(tf_models.load_model(
            _CD_MODEL_DIR / f"Model_{i}seed" / f"Model_{i}seed.h5", compile=False))
        scalers_cd.append(str(_CD_MODEL_DIR / f"Model_{i}seed" / "scalers.json"))
        models_rt.append(tf_models.load_model(
            _RT_MODEL_DIR / f"Model_{i}seed" / f"Model_{i}seed.h5", compile=False))
        scalers_rt.append(str(_RT_MODEL_DIR / f"Model_{i}seed" / "scalers.json"))

# ---------------------------------------------------------------------------
# Calcular water-plot completo (CD y R_total)
# ---------------------------------------------------------------------------
modo_str = "TMM" if USE_TMM else "NN"
print(f"Par: {PAIR}  |  d1={D1_NM} nm  d2={D2_NM} nm  |  {modo_str}")
print(f"Calculando water-plot ({len(thetas_all)} ángulos)...")

CD_matrix = np.zeros((len(thetas_all), N_FREQS))
R_matrix  = np.zeros((len(thetas_all), N_FREQS))
Rr_matrix = np.zeros((len(thetas_all), N_FREQS))
Rl_matrix = np.zeros((len(thetas_all), N_FREQS))

if USE_TMM:
    for idx, theta in enumerate(thetas_all):
        phi_rad   = np.deg2rad(theta)
        l1, l2    = _build_layers(PAIR, D1_NM * 1e-9, D2_NM * 1e-9, phi_rad)
        structure = LayeredStructure(superstrate=Air(), substrate=Au(), layers=[l1, l2])
        results   = [calculate_circular_dichroism_ref(f, 0, structure) for f in FREQS]
        CD_matrix[idx] = np.array([abs(r[1]) for r in results])
        R_matrix[idx]  = np.array([r[2]     for r in results])
        Rr_matrix[idx] = np.array([r[3]     for r in results])
        Rl_matrix[idx] = np.array([r[4]     for r in results])
else:
    for idx, theta in enumerate(thetas_all):
        params_batch = np.column_stack([
            np.full(N_FREQS, theta), np.full(N_FREQS, D1_NM),
            np.full(N_FREQS, D2_NM), FREQS,
        ])
        preds_cd, preds_rt = [], []
        for m, sp in zip(models_cd, scalers_cd):
            batch, _ = auxf.predict(m, params_batch, _DATABASE_CD, scaler_path=sp)
            preds_cd.append([abs(float(np.squeeze(v))) for v in batch])
        for m, sp in zip(models_rt, scalers_rt):
            batch, _ = auxf.predict(m, params_batch, _DATABASE_RT, scaler_path=sp)
            preds_rt.append([abs(float(np.squeeze(v))) for v in batch])
        CD_matrix[idx] = np.mean(preds_cd, axis=0)
        R_matrix[idx]  = np.mean(preds_rt, axis=0)
        # R_r y R_l no disponibles en modo NN

# ---------------------------------------------------------------------------
# Encontrar la cresta automáticamente en el rango PHI_MIN-PHI_MAX
# ---------------------------------------------------------------------------
mask_phi   = (thetas_all >= PHI_MIN) & (thetas_all <= PHI_MAX)
thetas_sel = thetas_all[mask_phi]
CD_sel     = CD_matrix[mask_phi]
R_sel      = R_matrix[mask_phi]

ridge_idx  = np.argmax(CD_sel, axis=1)
freq_ridge = FREQS[ridge_idx]
cd_ridge   = CD_sel[np.arange(len(thetas_sel)), ridge_idx]
r_ridge    = R_sel[np.arange(len(thetas_sel)), ridge_idx]

Rr_sel  = Rr_matrix[mask_phi]
Rl_sel  = Rl_matrix[mask_phi]
rr_ridge = Rr_sel[np.arange(len(thetas_sel)), ridge_idx]
rl_ridge = Rl_sel[np.arange(len(thetas_sel)), ridge_idx]

# Ajuste lineal a la cresta
coeffs   = np.polyfit(thetas_sel, freq_ridge, 1)
freq_fit = np.polyval(coeffs, thetas_sel)

print(f"\nCresta encontrada:")
print(f"  phi={thetas_sel[0]:.0f}° → {freq_ridge[0]:.0f} cm⁻¹  (CD={cd_ridge[0]:.2f}  R={r_ridge[0]:.2f})")
print(f"  phi={thetas_sel[-1]:.0f}° → {freq_ridge[-1]:.0f} cm⁻¹  (CD={cd_ridge[-1]:.2f}  R={r_ridge[-1]:.2f})")
print(f"  Ajuste lineal: freq = {coeffs[0]:.2f}·phi + {coeffs[1]:.1f} cm⁻¹")
print(f"  CD_max={cd_ridge.max():.3f}  CD_min={cd_ridge.min():.3f}  CD_medio={cd_ridge.mean():.3f}")
print(f"  R_max={r_ridge.max():.3f}   R_min={r_ridge.min():.3f}   R_medio={r_ridge.mean():.3f}")
print(f"  Rango espectral: {freq_ridge.min():.0f}–{freq_ridge.max():.0f} cm⁻¹  "
      f"({1e4/freq_ridge.max():.1f}–{1e4/freq_ridge.min():.1f} µm)")

# ---------------------------------------------------------------------------
# Figura: 3 paneles
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

# --- Panel 1: water-plot + cresta ---
vmax = np.max(CD_matrix)
im   = ax1.pcolormesh(FREQS, thetas_all, CD_matrix,
                      cmap="inferno", shading="auto", vmin=0, vmax=vmax)
fig.colorbar(im, ax=ax1, label=f"|CD| ({modo_str})")
ax1.plot(freq_ridge, thetas_sel, "w-",  lw=2,   label="cresta real")
ax1.plot(freq_fit,   thetas_sel, "c--", lw=1.5, label=f"ajuste lineal")
ax1.set_xlabel("Número de onda (cm⁻¹)", fontsize=12)
ax1.set_ylabel("φ (°)", fontsize=12)
ax1.set_yticks(np.arange(0, 181, 30))
ax1.set_title(f"Water-plot + cresta\n{PAIR.replace('_','/')}  d₁={D1_NM} d₂={D2_NM} nm", fontsize=11)
ax1.legend(fontsize=9)

def _add_freq_axis(ax, thetas_sel, freq_ridge):
    ax2b = ax.twiny()
    ax2b.set_xlim(ax.get_xlim())
    phi_ticks  = np.linspace(thetas_sel[0], thetas_sel[-1], 6)
    freq_ticks = np.interp(phi_ticks, thetas_sel, freq_ridge)
    ax2b.set_xticks(phi_ticks)
    ax2b.set_xticklabels([f"{f:.0f}" for f in freq_ticks])
    ax2b.set_xlabel("Frecuencia (cm⁻¹)", fontsize=11)

# --- Panel 2: CD a lo largo de la cresta ---
ax2.plot(thetas_sel, cd_ridge, "o-", ms=4, color="tab:orange", lw=1.5)
ax2.set_xlabel("φ (°)", fontsize=12)
ax2.set_ylabel("|CD|", fontsize=12)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.axhline(cd_ridge.mean(), color="gray", ls="--", lw=1,
            label=f"CD medio = {cd_ridge.mean():.2f}")
ax2.legend(fontsize=10)
ax2.set_title(
    f"CD en la cresta\nCD_max={cd_ridge.max():.2f}  CD_min={cd_ridge.min():.2f}",
    fontsize=11)
_add_freq_axis(ax2, thetas_sel, freq_ridge)

# --- Panel 3: Reflectancia (R_r, R_l, R_total) a lo largo de la cresta ---
if USE_TMM:
    ax3.plot(thetas_sel, rr_ridge, "o-", ms=4, color="tab:red",   lw=1.5, label="$R_r$")
    ax3.plot(thetas_sel, rl_ridge, "o-", ms=4, color="tab:blue",  lw=1.5, label="$R_l$")
    ax3.plot(thetas_sel, r_ridge,  "o-", ms=4, color="tab:green", lw=1.5, label="$R_{total}$", alpha=0.7)
else:
    ax3.plot(thetas_sel, r_ridge, "o-", ms=4, color="tab:blue", lw=1.5, label="$R_{total}$")
ax3.set_xlabel("φ (°)", fontsize=12)
ax3.set_ylabel("Reflectancia", fontsize=12)
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_title(
    f"Reflectancia en la cresta\nR_r_max={rr_ridge.max():.2f}  R_l_max={rl_ridge.max():.2f}" if USE_TMM else
    f"Reflectancia en la cresta\nR_max={r_ridge.max():.2f}  R_min={r_ridge.min():.2f}",
    fontsize=11)
_add_freq_axis(ax3, thetas_sel, freq_ridge)

fig.tight_layout()

out = Path(__file__).parent / f"ridge_{PAIR}_d1{D1_NM}_d2{D2_NM}.png"
fig.savefig(out, dpi=150)
print(f"\nGuardado: {out.name}")

plt.show()