"""
===========================================================
Prediction
===========================================================
This script contains the functions to run inference with a trained model,
including loading the model, running predictions,
selecting best candidates using an oracle distance metric, and saving comparison plots.

Author: [Lucia F. Alvarez-Tomillo]
Date: [xx/xx/2025]
"""

import os
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
    Air,
    Au,
    MoO3,
    LayeredStructure,
    calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf


def run_inference(
    model_dir="./Forward_Models_Trained_bilayers_MoO3",
    database="./Database_MoO3_bilayers",
    num_seeds=5,
    N_pred=2,
    superstrate_mat=Air(),
    substrate_mat=Au(),
    alpha=0,
):
    """Load an ensemble of trained models and produce CD spectra."""
    t1 = time.time()

    models_list, scaler_paths = [], []
    for i in range(1, num_seeds + 1):
        model_path = Path(model_dir) / f"Model_{i}seed" / f"Model_{i}seed.h5"
        scaler_path = Path(model_dir) / f"Model_{i}seed" / "scalers.json"
        models_list.append(models.load_model(model_path, compile=False))
        scaler_paths.append(scaler_path)

    results_dir = Path(model_dir) / "results"
    os.makedirs(results_dir, exist_ok=True)

    for i in range(N_pred):
        freqs = np.linspace(600, 1100, 1000)
        d_layers_nm = np.random.randint(200, 2000 + 1, (2,))
        angles_deg = np.random.randint(0, 180 + 1, (1,))

        print(f"\nConfiguración {i+1}: d_layers_nm={d_layers_nm}, angles_deg={angles_deg}")

        structure = LayeredStructure(
            superstrate=superstrate_mat,
            substrate=substrate_mat,
            layers=[
                MoO3(d=d_layers_nm[0] * 1e-9, phi=np.deg2rad(angles_deg[0])),
                MoO3(d=d_layers_nm[1] * 1e-9),
            ],
        )

        CD_true = np.array([
            abs(calculate_circular_dichroism_ref(f, alpha, structure)[1]) for f in freqs
        ])

        parameters_batch = np.array([
            np.concatenate((angles_deg, d_layers_nm, np.array([f]))) for f in freqs
        ])

        preds = []
        for m, sp in zip(models_list, scaler_paths):
            CD_pred_batch, _ = auxf.predict(m, parameters_batch, database, scaler_path=sp)
            preds.append([abs(float(np.squeeze(cd))) for cd in CD_pred_batch])
        CD_pred = np.mean(preds, axis=0)

        lambda_nm = 1e4 / freqs
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(lambda_nm, CD_true, color="#0b0b0b", lw=2.0, label="TMM")
        ax.plot(lambda_nm, CD_pred, color="#2a78d6", lw=1.6, ls="--", label="NN")
        ax.set_xlabel(r"$\lambda$ ($\mu m$)")
        ax.set_ylabel(r"$|\mathrm{CD}|$")
        plt.legend()
        plt.savefig(results_dir / f"comparison_parameters{[d_layers_nm, angles_deg]}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Total execution time: {time.time() - t1:.2f} seconds")


if __name__ == "__main__":
    run_inference(
        model_dir=str(ROOT_PATH / "Models" / "CD" / "MoO3_MoO3"),
        database=str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MoO3"),
        num_seeds=5,
        N_pred=10,
        superstrate_mat=Air(),
        substrate_mat=Au(),
        alpha=0,
    )