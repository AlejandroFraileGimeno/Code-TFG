# -*- coding: utf-8 -*-
"""
Water-plot de CD vs phi (0-180°) a d1, d2 fijos.

Edita la sección CONFIG antes de ejecutar.
  USE_TMM = False  →  NN  (segundos)
  USE_TMM = True   →  TMM (minutos, ground truth)
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — edita aquí antes de ejecutar
# ============================================================
PAIR       = "MoO3_V2O5"   # par de materiales
D1_NM      = 400                # espesor capa 1 (nm)
D2_NM      = 900               # espesor capa 2 (nm)
THETA_STEP = 0.1                  # paso en grados (1 = 181 puntos)
USE_TMM    = True              # False = NN (rápido)  |  True = TMM (exacto)
NUM_SEEDS  = 1                  # solo para modo NN
# ============================================================

# Pares disponibles:
#   "MgTeMoO6_MgTeMoO6"  "MoO3_MgTeMoO6"  "MoO3_MoO3"
#   "MoO3_V2O5"          "V2O5_V2O5"       "V2O5_MgTeMoO6"
#   "MgTeMoO6_MoO3"      "MgTeMoO6_V2O5"   "V2O5_MoO3"

ROOT_PATH = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT_PATH / "TMM"))
from generalized_transfer_matrix_method import (
    Air, Au, MgTeMoO6, MoO3, V2O5, LayeredStructure,
    calculate_circular_dichroism_ref,
)

# Mapeo par → (surrogate, capa1(d_m, phi_rad), capa2(d_m))
def _layers(pair, d1_m, d2_m, phi_rad):
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

_SURROGATE_MAP = {
    "MgTeMoO6_MgTeMoO6": "MgTeMoO6_MgTeMoO6",
    "MoO3_MgTeMoO6":     "MoO3_MgTeMoO6",
    "MoO3_MoO3":         "MoO3_MoO3",
    "MoO3_V2O5":         "MoO3_V2O5",
    "V2O5_V2O5":         "V2O5_V2O5",
    "V2O5_MgTeMoO6":     "V2O5_MgTeMoO6",
    "MgTeMoO6_MoO3":     "MgTeMoO6_MoO3",
    "MgTeMoO6_V2O5":     "MgTeMoO6_V2O5",
    "V2O5_MoO3":         "V2O5_MoO3",
}

if PAIR not in _SURROGATE_MAP:
    print(f"Par '{PAIR}' no reconocido. Opciones: {list(_SURROGATE_MAP)}")
    sys.exit(1)

# Rango de frecuencias del modelo
_CD_MODEL_DIR = ROOT_PATH / "Models" / "CD" / PAIR
_scaler_ref   = json.load(open(_CD_MODEL_DIR / "Model_1seed" / "scalers.json"))
FREQ_MIN = _scaler_ref["feature_min"][3]
FREQ_MAX = _scaler_ref["feature_max"][3]
N_FREQS  = _scaler_ref["n_freqs"]
FREQS    = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

thetas = np.arange(0, 180 + THETA_STEP, THETA_STEP)
print(f"Par: {PAIR}  |  d1={D1_NM} nm  d2={D2_NM} nm")
print(f"Modo: {'TMM (exacto)' if USE_TMM else 'NN (rápido)'}  |  {len(thetas)} ángulos\n")

# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
CD_matrix = np.zeros((len(thetas), N_FREQS))

if USE_TMM:
    import time
    t0 = time.time()
    for idx, theta in enumerate(thetas):
        phi_rad = np.deg2rad(theta)
        l1, l2  = _layers(PAIR, D1_NM * 1e-9, D2_NM * 1e-9, phi_rad)
        structure = LayeredStructure(
            superstrate=Air(), substrate=Au(), layers=[l1, l2]
        )
        results = [calculate_circular_dichroism_ref(f, 0, structure) for f in FREQS]
        CD_matrix[idx] = np.array([abs(r[1]) for r in results])
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (idx + 1) * (len(thetas) - idx - 1)
            print(f"  {idx+1}/{len(thetas)}  (~{eta/60:.1f} min restantes)")
    print(f"TMM completado en {(time.time()-t0)/60:.1f} min\n")

else:
    surrogate_folder = _SURROGATE_MAP[PAIR]
    sys.path.insert(0, str(ROOT_PATH / "Surrogates" / "CD" / surrogate_folder))
    import utils_nn_forward as auxf
    from tensorflow.keras import models as tf_models
    _DATABASE = str(ROOT_PATH / "Datasets" / "CD" / PAIR)

    print(f"Cargando {NUM_SEEDS} modelo(s) CD...")
    models_cd, scalers_cd = [], []
    for i in range(1, NUM_SEEDS + 1):
        models_cd.append(tf_models.load_model(
            _CD_MODEL_DIR / f"Model_{i}seed" / f"Model_{i}seed.h5", compile=False
        ))
        scalers_cd.append(str(_CD_MODEL_DIR / f"Model_{i}seed" / "scalers.json"))
    print("Modelos cargados.\n")

    print(f"Calculando {len(thetas)} ángulos...")
    for idx, theta in enumerate(thetas):
        params_batch = np.column_stack([
            np.full(N_FREQS, theta),
            np.full(N_FREQS, D1_NM),
            np.full(N_FREQS, D2_NM),
            FREQS,
        ])
        preds = []
        for m, sp in zip(models_cd, scalers_cd):
            batch, _ = auxf.predict(m, params_batch, _DATABASE, scaler_path=sp)
            preds.append([abs(float(np.squeeze(v))) for v in batch])
        CD_matrix[idx] = np.mean(preds, axis=0)
    print("Listo.\n")

# ---------------------------------------------------------------------------
# Plot — estilo TFG
# ---------------------------------------------------------------------------
modo_str = "TMM" if USE_TMM else "NN"

plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           13,
    "axes.labelsize":      14,
    "axes.titlesize":      13,
    "xtick.labelsize":     12,
    "ytick.labelsize":     12,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
})

fig, ax = plt.subplots(figsize=(9, 5.5))
vmax = np.max(CD_matrix)
im   = ax.pcolormesh(
    FREQS, thetas, CD_matrix,
    cmap="inferno", shading="auto",
    vmin=0, vmax=vmax,
    rasterized=True,
)
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label(rf"$|\mathrm{{CD}}|$ normalizado ({modo_str})")
cbar.outline.set_linewidth(0.9)

ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
ax.set_ylabel(r"$\varphi$ (°)")
ax.set_yticks(np.arange(0, 181, 30))
ax.set_title(
    f"Water-plot CD — {PAIR.replace('_', '/')}   "
    rf"$d_1$={D1_NM:.0f} nm  $d_2$={D2_NM:.0f} nm  [{modo_str}]",
)
fig.tight_layout()

out = Path(__file__).parent / f"theta_sweep_{PAIR}_d1{D1_NM:.0f}_d2{D2_NM:.0f}_{modo_str}.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Guardado: {out.name}")

plt.show()