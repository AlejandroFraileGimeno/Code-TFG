# -*- coding: utf-8 -*-
"""
Comparativa espectro objetivo vs reconstruido (red inversa) — una muestra por par.
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

COLOR_OBJ   = "#0b0b0b"
COLOR_RECON = "#2a78d6"
N_TRAIN     = 20_000
SEED_PLOT   = 7

MODELS_DIR  = ROOT / "Models"   / "T_xx"
DATASET_DIR = ROOT / "Datasets" / "T_xx"
TANDEM_DIR  = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuración por par de materiales
# ---------------------------------------------------------------------------
def make_tmm_moo3_v2o5(th1, th2, d1, d2, freqs):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[MoO3(d=d1*1e-9, phi=np.deg2rad(th1)),
                V2O5(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0]) for f in freqs])

def make_tmm_v2o5_moo3(th1, th2, d1, d2, freqs):
    s = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[V2O5(d=d1*1e-9, phi=np.deg2rad(th1)),
                MoO3(d=d2*1e-9, phi=np.deg2rad(th2))],
    )
    return np.array([float(calculate_transmission(f, 0, s, basis="linear")[0]) for f in freqs])

def load_test_spectrum_csv(dataset_dir, rng):
    T_all = np.loadtxt(dataset_dir / "T_xx_spectra.csv", delimiter=",").astype(np.float32)
    idx   = rng.choice(np.arange(N_TRAIN * 2, len(T_all)))
    return T_all[idx]

def load_test_spectrum_chunks(dataset_dir, rng):
    chunks_dir = dataset_dir / "tmp_chunks"
    # Elegir un chunk bien alejado del rango de entrenamiento
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    chunk_file  = rng.choice(chunk_files[100:])   # ≥ chunk 100 → ≥ 50 000 muestras
    T_chunk = np.load(chunk_file)["T_xx"]
    return T_chunk[rng.integers(len(T_chunk))]

PARES = [
    (
        r"MoO$_3$ / V$_2$O$_5$",
        MODELS_DIR  / "MoO3_V2O5_BaF2" / f"Inverse_N{N_TRAIN}",
        DATASET_DIR / "MoO3_V2O5_BaF2",
        load_test_spectrum_csv,
        make_tmm_moo3_v2o5,
        TANDEM_DIR  / "MoO3_V2O5_BaF2" / "Evaluación" / "comparativa_tandem_single.png",
    ),
    (
        r"V$_2$O$_5$ / MoO$_3$",
        MODELS_DIR  / "V2O5_MoO3_BaF2" / f"Inverse_N{N_TRAIN}",
        DATASET_DIR / "V2O5_MoO3_BaF2",
        load_test_spectrum_chunks,
        make_tmm_v2o5_moo3,
        TANDEM_DIR  / "V2O5_MoO3_BaF2" / "Evaluación" / "comparativa_tandem_single.png",
    ),
]

# ---------------------------------------------------------------------------
# Generar una figura por par
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED_PLOT)

for titulo, inverse_dir, dataset_dir, loader_fn, tmm_fn, out_path in PARES:
    scalers   = json.loads((inverse_dir / "scalers.json").read_text())
    param_min = np.array(scalers["param_min"], dtype=np.float32)
    param_max = np.array(scalers["param_max"], dtype=np.float32)
    freqs     = np.linspace(scalers["freq_min"], scalers["freq_max"], scalers["n_freqs"])

    # Cargar modelo inverso
    inv_model = tf.keras.models.load_model(
        inverse_dir / "Model_1seed" / "inverse.keras", compile=False
    )

    # Espectro objetivo del conjunto de test
    target = loader_fn(dataset_dir, rng)

    # Predicción inversa → parámetros
    p_norm        = inv_model.predict(target.reshape(1, -1), verbose=0)[0]
    th1, th2, d1, d2 = p_norm * (param_max - param_min) + param_min

    # Espectro reconstruido con TMM
    T_recon = tmm_fn(th1, th2, d1, d2, freqs)

    # -----------------------------------------------------------------------
    # Figura
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot(freqs, target,  color=COLOR_OBJ,   lw=2.0,        label="Objetivo")
    ax.plot(freqs, T_recon, color=COLOR_RECON, lw=1.6, ls="--", label="Reconstruido")

    ax.set_title(
        titulo + "\n"
        + rf"$\phi_1={th1:.0f}^\circ,\;\phi_2={th2:.0f}^\circ,"
          rf"\;d_1={d1:.0f}\,\mathrm{{nm}},\;d_2={d2:.0f}\,\mathrm{{nm}}$",
        pad=8, fontsize=15,
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
    out_dir = Path(__file__).resolve().parents[2] / "Arreglos en Gráficos"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{out_path.stem}_{out_path.parts[-3]}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"Guardado: {out_file}")
    plt.show()
