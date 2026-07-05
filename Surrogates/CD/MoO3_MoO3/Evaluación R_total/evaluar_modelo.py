"""
===========================================================
Numerical evaluation — R_total surrogate
===========================================================
Computes R², MAE and RMSE between NN predictions and TMM
ground truth (R_total = R_r + R_l) over N random configurations.

Author: [Lucia F. Alvarez-Tomillo / Alejandro Fraile]
Date:   [xx/xx/2026]
"""

import sys
from pathlib import Path
import numpy as np
from tensorflow.keras import models

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf


def evaluate(model_dir, database, num_seeds=1, N=50, alpha=0):
    freqs = np.linspace(600, 1100, 1000)

    models_list, scaler_paths = [], []
    for i in range(1, num_seeds + 1):
        model_path = Path(model_dir) / f"Model_{i}seed" / f"Model_{i}seed.h5"
        scaler_path = Path(model_dir) / f"Model_{i}seed" / "scalers.json"
        models_list.append(models.load_model(model_path, compile=False))
        scaler_paths.append(scaler_path)

    r2_list, mae_list, rmse_list = [], [], []

    for i in range(N):
        d1, d2 = np.random.randint(200, 2001, size=2)
        theta = np.random.randint(0, 181)

        structure = LayeredStructure(
            superstrate=Air(),
            substrate=Au(),
            layers=[
                MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
                MoO3(d=d2 * 1e-9),
            ],
        )

        R_true = np.array([
            calculate_circular_dichroism_ref(f, alpha, structure)[2] for f in freqs
        ])

        params = np.column_stack([
            np.full(len(freqs), theta),
            np.full(len(freqs), d1),
            np.full(len(freqs), d2),
            freqs,
        ])

        preds = []
        for m, sp in zip(models_list, scaler_paths):
            R_pred_batch, _ = auxf.predict(m, params, database, scaler_path=sp)
            preds.append([float(np.squeeze(r)) for r in R_pred_batch])
        R_pred = np.mean(preds, axis=0)

        ss_res = np.sum((R_true - R_pred) ** 2)
        ss_tot = np.sum((R_true - np.mean(R_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        mae = np.mean(np.abs(R_true - R_pred))
        rmse = np.sqrt(np.mean((R_true - R_pred) ** 2))

        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)

        print(f"[{i+1:3d}/{N}] d1={d1:4d} d2={d2:4d} theta={theta:3d}° | R²={r2:.4f}  MAE={mae:.6f}  RMSE={rmse:.6f}")

    print(f"\n--- Promedio sobre {N} configuraciones ({num_seeds} modelos) ---")
    print(f"R²   = {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    print(f"MAE  = {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}")
    print(f"RMSE = {np.mean(rmse_list):.6f} ± {np.std(rmse_list):.6f}")


if __name__ == "__main__":
    evaluate(
        model_dir=str(ROOT_PATH / "Models" / "R_total" / "MoO3_MoO3"),
        database=str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MoO3"),
        num_seeds=5,
        N=50,
    )