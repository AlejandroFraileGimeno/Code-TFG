# -*- coding: utf-8 -*-
"""
Comparativa TMM vs surrogate — una estructura aleatoria (MoO3/MoO3).
Genera dos figuras con la misma estructura: |CD| y R_total.
Estilo publicación TFG (mismo formato que Tandem/T_xx/plot_forward_single.py).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.keras import models

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf

# ---------------------------------------------------------------------------
# Estilo TFG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           15,
    "axes.labelsize":      16,
    "axes.titlesize":      15,
    "xtick.labelsize":     13,
    "ytick.labelsize":     13,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     13,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

COLOR_TMM = "#0b0b0b"
COLOR_NN  = "#2a78d6"
SEED_PLOT = 21
D_MIN, D_MAX = 200, 2000
ALPHA = 0

BASE_DIR = Path(__file__).resolve().parent
DATABASE = str(ROOT / "Datasets" / "CD" / "MoO3_MoO3")

# ---------------------------------------------------------------------------
# Estructura aleatoria (la misma para |CD| y R_total)
# ---------------------------------------------------------------------------
np.random.seed(SEED_PLOT)
d1    = np.random.randint(D_MIN, D_MAX + 1)
d2    = np.random.randint(D_MIN, D_MAX + 1)
theta = np.random.randint(0, 181)

freqs = np.linspace(600, 1100, 1000)

structure = LayeredStructure(
    superstrate=Air(),
    substrate=Au(),
    layers=[
        MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
        MoO3(d=d2 * 1e-9),
    ],
)

cd_ref = [calculate_circular_dichroism_ref(f, ALPHA, structure) for f in freqs]
CD_true = np.array([abs(c[1]) for c in cd_ref])
R_true  = np.array([c[2] for c in cd_ref])

params = np.column_stack([
    np.full(len(freqs), theta),
    np.full(len(freqs), d1),
    np.full(len(freqs), d2),
    freqs,
])

# ---------------------------------------------------------------------------
# Casos: (etiqueta y, carpeta modelo, valores TMM, abs de la prediccion, salida)
# ---------------------------------------------------------------------------
CASOS = [
    (r"$|\mathrm{CD}|$",
     ROOT / "Models" / "CD" / "MoO3_MoO3" / "Model_1seed",
     CD_true, True,
     BASE_DIR / "Evaluación" / "comparativa_cd_single.png"),
    (r"$R_{\mathrm{total}}$",
     ROOT / "Models" / "R_total" / "MoO3_MoO3" / "Model_1seed",
     R_true, False,
     BASE_DIR / "Evaluación R_total" / "comparativa_r_total_single.png"),
]

for ylabel, model_dir, y_true, use_abs, out_path in CASOS:
    model = models.load_model(model_dir / "Model_1seed.h5", compile=False)
    scaler_path = model_dir / "scalers.json"

    pred_batch, _ = auxf.predict(model, params, DATABASE, scaler_path=scaler_path)
    y_pred = np.squeeze(np.asarray(pred_batch, dtype=float))
    if use_abs:
        y_pred = np.abs(y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot(freqs, y_true, color=COLOR_TMM, lw=2.0, label="TMM")
    ax.plot(freqs, y_pred, color=COLOR_NN,  lw=1.6, ls="--", label="Surrogate")

    ax.set_title(
        r"MoO$_3$ / MoO$_3$" + "\n"
        + rf"$\theta={theta}^\circ,\;d_1={d1}\,\mathrm{{nm}},\;d_2={d2}\,\mathrm{{nm}}$",
        pad=8, fontsize=13,
    )

    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ax.grid(True, which="both")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.close(fig)
