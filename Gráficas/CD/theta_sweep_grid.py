# -*- coding: utf-8 -*-
"""
Grid de water-plots CD vs phi para diferentes combinaciones de d1/d2.
Genera todos los subplots en una sola figura para comparar visualmente.

Edita la sección CONFIG antes de ejecutar.
"""

import sys
import json
import time
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — edita aquí antes de ejecutar
# ============================================================
PAIR       = "MoO3_V2O5"    # par de materiales
USE_TMM    = True            # True = TMM (exacto) | False = NN (rápido)
NUM_SEEDS  = 1               # solo para modo NN
THETA_STEP = 2               # paso en grados (2 = 91 puntos, más rápido)

# Combinaciones (d1_nm, d2_nm) a comparar
COMBOS = [
    (400,  800),
    (500,  700),
    (500,  900),
    (400,  900),
]
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

if PAIR not in _SURROGATE_MAP:
    print(f"Par '{PAIR}' no reconocido.")
    sys.exit(1)

# Rango de frecuencias
_CD_MODEL_DIR = ROOT_PATH / "Models" / "CD" / PAIR
_scaler_ref   = json.load(open(_CD_MODEL_DIR / "Model_1seed" / "scalers.json"))
FREQ_MIN = _scaler_ref["feature_min"][3]
FREQ_MAX = _scaler_ref["feature_max"][3]
N_FREQS  = _scaler_ref["n_freqs"]
FREQS    = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
thetas   = np.arange(0, 180 + THETA_STEP, THETA_STEP)

# ---------------------------------------------------------------------------
# Cargar modelos NN si hace falta
# ---------------------------------------------------------------------------
if not USE_TMM:
    sys.path.insert(0, str(ROOT_PATH / "Surrogates" / "CD" / _SURROGATE_MAP[PAIR]))
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

# ---------------------------------------------------------------------------
# Calcular matriz CD para cada combo
# ---------------------------------------------------------------------------
def compute_cd_matrix(d1_nm, d2_nm):
    mat = np.zeros((len(thetas), N_FREQS))
    if USE_TMM:
        for idx, theta in enumerate(thetas):
            phi_rad = np.deg2rad(theta)
            l1, l2  = _build_layers(PAIR, d1_nm * 1e-9, d2_nm * 1e-9, phi_rad)
            structure = LayeredStructure(superstrate=Air(), substrate=Au(), layers=[l1, l2])
            results   = [calculate_circular_dichroism_ref(f, 0, structure) for f in FREQS]
            mat[idx]  = np.array([abs(r[1]) for r in results])
    else:
        for idx, theta in enumerate(thetas):
            params_batch = np.column_stack([
                np.full(N_FREQS, theta),
                np.full(N_FREQS, d1_nm),
                np.full(N_FREQS, d2_nm),
                FREQS,
            ])
            preds = []
            for m, sp in zip(models_cd, scalers_cd):
                batch, _ = auxf.predict(m, params_batch, _DATABASE, scaler_path=sp)
                preds.append([abs(float(np.squeeze(v))) for v in batch])
            mat[idx] = np.mean(preds, axis=0)
    return mat

modo_str = "TMM" if USE_TMM else "NN"
n_combos = len(COMBOS)
ncols    = 2
nrows    = math.ceil(n_combos / ncols)

print(f"Par: {PAIR}  |  Modo: {modo_str}  |  {len(thetas)} ángulos  |  {n_combos} combos\n")
t_total = time.time()

matrices = []
for i, (d1, d2) in enumerate(COMBOS):
    print(f"[{i+1}/{n_combos}]  d1={d1} nm  d2={d2} nm ...")
    t0  = time.time()
    mat = compute_cd_matrix(d1, d2)
    matrices.append(mat)
    print(f"       listo en {time.time()-t0:.0f}s  |  CD_max={mat.max():.3f}")

print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")

# ---------------------------------------------------------------------------
# Figura grid — estilo TFG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           12,
    "axes.labelsize":      12,
    "axes.titlesize":      12,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
})

vmax_global = max(m.max() for m in matrices)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)

for i, (ax, mat, (d1, d2)) in enumerate(zip(axes.flat, matrices, COMBOS)):
    im = ax.pcolormesh(
        FREQS, thetas, mat,
        cmap="inferno", shading="auto",
        vmin=0, vmax=vmax_global,
        rasterized=True,
    )
    ax.set_title(rf"$d_1$={d1} nm  $d_2$={d2} nm   "
                 rf"$|\mathrm{{CD}}|_\mathrm{{max}}$={mat.max():.2f}")
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel(r"$\varphi$ (°)")
    ax.set_yticks(np.arange(0, 181, 30))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(rf"$|\mathrm{{CD}}|$ ({modo_str})")

# Ocultar subplots sobrantes si n_combos es impar
for j in range(i + 1, nrows * ncols):
    axes.flat[j].set_visible(False)

fig.suptitle(
    f"Comparativa water-plot CD — {PAIR.replace('_', '/')}  [{modo_str}]",
    fontsize=13, y=1.01,
)
fig.tight_layout()

out = Path(__file__).parent / f"theta_grid_{PAIR}_{modo_str}.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Guardado: {out.name}")

plt.show()