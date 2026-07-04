"""
Prediction — CD_norm surrogate vs TMM (MoO3 / MgTeMoO6)
"""

import sys
import time
from pathlib import Path
import numpy as np
from tensorflow.keras import models
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

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, MgTeMoO6, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf


def run_inference(model_dir, database, num_seeds=5, N_pred=5, alpha=0):
    t1 = time.time()

    models_list, scaler_paths = [], []
    for i in range(1, num_seeds + 1):
        model_path  = Path(model_dir) / f"Model_{i}seed" / f"Model_{i}seed.h5"
        scaler_path = Path(model_dir) / f"Model_{i}seed" / "scalers.json"
        models_list.append(models.load_model(model_path, compile=False))
        scaler_paths.append(scaler_path)

    results_dir = Path(model_dir) / "results"
    results_dir.mkdir(exist_ok=True)

    freqs     = np.linspace(600, 1100, 1000)
    lambda_mu = 1e4 / freqs

    for i in range(N_pred):
        d1, d2 = np.random.randint(200, 2001, size=2)
        theta   = np.random.randint(0, 181)
        print(f"\nConfiguracion {i+1}: d1={d1} nm  d2={d2} nm  theta={theta} deg")

        structure = LayeredStructure(
            superstrate=Air(), substrate=Au(),
            layers=[
                MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
                MgTeMoO6(d=d2 * 1e-9),
            ],
        )

        CD_true = np.array([
            abs(calculate_circular_dichroism_ref(f, alpha, structure)[1]) for f in freqs
        ])

        params = np.column_stack([
            np.full(len(freqs), theta), np.full(len(freqs), d1),
            np.full(len(freqs), d2), freqs,
        ])

        preds = []
        for m, sp in zip(models_list, scaler_paths):
            CD_pred_batch, _ = auxf.predict(m, params, database, scaler_path=sp)
            preds.append([abs(float(np.squeeze(cd))) for cd in CD_pred_batch])
        CD_pred = np.mean(preds, axis=0)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(lambda_mu, CD_true, color="#0b0b0b", lw=2.0, label="TMM")
        ax.plot(lambda_mu, CD_pred, color="#2a78d6", lw=1.6, ls="--", label="NN")
        ax.set_xlabel(r"$\lambda$ ($\mu$m)")
        ax.set_ylabel(r"$|\mathrm{CD}|$")
        ax.set_title(rf"MoO3/MgTeMoO6 — $d_1$={d1} nm  $d_2$={d2} nm  $\theta$={theta}°")
        ax.legend()
        fig.tight_layout()
        fname = results_dir / f"comparison_d1{d1}_d2{d2}_th{theta}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {fname.relative_to(ROOT_PATH)}")

    print(f"\nTiempo total: {time.time() - t1:.2f} s")


if __name__ == "__main__":
    run_inference(
        model_dir=str(ROOT_PATH / "Models" / "CD" / "MoO3_MgTeMoO6"),
        database=str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MgTeMoO6"),
        num_seeds=5, N_pred=5,
    )