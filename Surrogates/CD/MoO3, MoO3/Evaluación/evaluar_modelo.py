"""
===========================================================
Numerical evaluation
===========================================================
Computes R², MAE and RMSE between NN predictions and TMM
ground truth over N random configurations. No plots.

Author: [Lucia F. Alvarez-Tomillo / Alejandro Fraile]
Date: [xx/xx/2026]
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


def evaluate(model_dir, database, num_seed=1, N=50, alpha=0):
    freqs = np.linspace(600, 1100, 1000)
    model_path = Path(model_dir) / f"Model_{num_seed}seed" / f"Model_{num_seed}seed.h5"
    scaler_path = Path(model_dir) / f"Model_{num_seed}seed" / "scalers.json"

    model = models.load_model(model_path, compile=False)

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

        CD_true = np.array([
            abs(calculate_circular_dichroism_ref(f, alpha, structure)[1]) for f in freqs
        ])

        params = np.column_stack([
            np.full(len(freqs), theta),
            np.full(len(freqs), d1),
            np.full(len(freqs), d2),
            freqs,
        ])
        CD_pred_batch, _ = auxf.predict(model, params, database, scaler_path=scaler_path)
        CD_pred = np.array([abs(float(np.squeeze(cd))) for cd in CD_pred_batch])

        ss_res = np.sum((CD_true - CD_pred) ** 2)
        ss_tot = np.sum((CD_true - np.mean(CD_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        mae = np.mean(np.abs(CD_true - CD_pred))
        rmse = np.sqrt(np.mean((CD_true - CD_pred) ** 2))

        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)

        print(f"[{i+1:3d}/{N}] d1={d1:4d} d2={d2:4d} theta={theta:3d}° | R²={r2:.4f}  MAE={mae:.6f}  RMSE={rmse:.6f}")

    print(f"\n--- Promedio sobre {N} configuraciones ---")
    print(f"R²   = {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    print(f"MAE  = {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}")
    print(f"RMSE = {np.mean(rmse_list):.6f} ± {np.std(rmse_list):.6f}")


if __name__ == "__main__":
    evaluate(
        model_dir=str(ROOT_PATH / "Models" / "CD" / "MoO3_MoO3"),
        database=str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MoO3"),
        num_seed=1,
        N=50,
    )