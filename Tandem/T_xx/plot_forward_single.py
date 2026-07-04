# -*- coding: utf-8 -*-
"""
Comparativa TMM vs surrogate forward — una estructura aleatoria por par de materiales.
Estilo publicación TFG.
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, V2O5, LayeredStructure, calculate_transmission,
)

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

COLOR_TMM  = "#0b0b0b"
COLOR_NN   = "#2a78d6"
ALPHA_BAND = 0.18
NUM_SEEDS  = 3
SEED_PLOT  = 8
D_MIN, D_MAX = 200, 1200

MODELS_DIR = ROOT / "Models" / "T_xx"
TANDEM_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuración por par de materiales
# ---------------------------------------------------------------------------
def make_layers_moo3_v2o5(d1, th1, d2, th2):
    return [MoO3(d=d1 * 1e-9, phi=np.deg2rad(th1)),
            V2O5(d=d2 * 1e-9, phi=np.deg2rad(th2))]

def make_layers_v2o5_moo3(d1, th1, d2, th2):
    return [V2O5(d=d1 * 1e-9, phi=np.deg2rad(th1)),
            MoO3(d=d2 * 1e-9, phi=np.deg2rad(th2))]

PARES = [
    (
        r"MoO$_3$ / V$_2$O$_5$",
        MODELS_DIR / "MoO3_V2O5_BaF2" / "Forward",
        make_layers_moo3_v2o5,
        TANDEM_DIR / "MoO3_V2O5_BaF2" / "Evaluación" / "comparativa_forward_single.png",
    ),
    (
        r"V$_2$O$_5$ / MoO$_3$",
        MODELS_DIR / "V2O5_MoO3_BaF2" / "Forward",
        make_layers_v2o5_moo3,
        TANDEM_DIR / "V2O5_MoO3_BaF2" / "Evaluación" / "comparativa_forward_single.png",
    ),
]

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def load_models(model_dir):
    return [
        tf.keras.models.load_model(model_dir / f"Model_{i}seed" / "forward.keras", compile=False)
        for i in range(1, NUM_SEEDS + 1)
    ]

def predict_ensemble(models, params_norm):
    preds = np.stack([m.predict(params_norm, verbose=0)[0] for m in models])
    return preds.mean(0), preds.std(0)

def tmm_spectrum(freqs, make_layers, th1, th2, d1, d2):
    structure = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=make_layers(d1, th1, d2, th2),
    )
    return np.array([float(calculate_transmission(f, 0, structure, basis="linear")[0])
                     for f in freqs])

# ---------------------------------------------------------------------------
# Generar una figura por par
# ---------------------------------------------------------------------------
np.random.seed(SEED_PLOT)

for titulo, model_dir, make_layers, out_path in PARES:
    scalers   = json.loads((model_dir / "scalers.json").read_text())
    param_min = np.array(scalers["param_min"], dtype=np.float32)
    param_max = np.array(scalers["param_max"], dtype=np.float32)
    freqs     = np.linspace(scalers["freq_min"], scalers["freq_max"], scalers["n_freqs"])

    models = load_models(model_dir)

    # Estructura aleatoria
    th1 = np.random.uniform(0,     180)
    th2 = np.random.uniform(0,     180)
    d1  = np.random.uniform(D_MIN, D_MAX)
    d2  = np.random.uniform(D_MIN, D_MAX)

    # TMM
    T_tmm = tmm_spectrum(freqs, make_layers, th1, th2, d1, d2)

    # NN ensemble
    p      = np.array([[th1, th2, d1, d2]], dtype=np.float32)
    p_norm = (p - param_min) / (param_max - param_min)
    T_nn, T_nn_std = predict_ensemble(models, p_norm)

    # -----------------------------------------------------------------------
    # Figura
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.fill_between(freqs, T_nn - T_nn_std, T_nn + T_nn_std,
                    color=COLOR_NN, alpha=ALPHA_BAND)
    ax.plot(freqs, T_tmm, color=COLOR_TMM, lw=2.0, label="Simulación")
    ax.plot(freqs, T_nn,  color=COLOR_NN,  lw=1.6, ls="--", label="Surrogate")

    # Parámetros en el título (dos líneas)
    ax.set_title(
        titulo + "\n"
        + rf"$\theta_1={th1:.0f}^\circ,\;\theta_2={th2:.0f}^\circ,\;d_1={d1:.0f}\,\mathrm{{nm}},\;d_2={d2:.0f}\,\mathrm{{nm}}$",
        pad=8, fontsize=13,
    )

    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel(r"$T_{xx}$")
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, which="both")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Guardado: {out_path}")
    plt.show()
