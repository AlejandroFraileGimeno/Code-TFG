"""
===========================================================
Inferencia T_ss  (NN vs TMM)
===========================================================
Carga el modelo entrenado y, para varias estructuras aleatorias, compara el
espectro de transmitancia T_ss predicho por la red con el cálculo físico exacto
(TMM). Guarda las gráficas de comparación en NN_Code/.../results/.
"""

import os
import time
from pathlib import Path
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt

from generalized_transfer_matrix_method import (
    Air,
    BaF2,
    MoO3,
    LayeredStructure,
    calculate_transmission,
)
import utils_nn_forward as auxf


def run_inference(
    model_dir="./Tss_Models_Trained_bilayers_MoO3",
    database="./Dataset_Tss_Bilayer",
    num_seed=1,
    N_pred=3,
    superstrate_mat=Air(),
    substrate_mat=BaF2(),
    alpha=np.deg2rad(40),
):
    """Carga un modelo entrenado y compara T_ss (red) vs T_ss (TMM)."""
    t1 = time.time()

    model_path = Path(model_dir) / f"Model_{num_seed}seed" / f"Model_{num_seed}seed.h5"
    scaler_path = Path(model_dir) / f"Model_{num_seed}seed" / "scalers.json"
    results_dir = Path(model_dir) / "results"
    os.makedirs(results_dir, exist_ok=True)

    model = models.load_model(str(model_path), compile=False)
    print("Modelo:")
    model.summary()

    for i in range(N_pred):
        freqs = np.linspace(600, 1100, 1000)
        # Solo estructuras FISICAS (T_ss<=1): si el TMM explota, probamos otra.
        while True:
            d_layers_nm = np.random.randint(200, 2000 + 1, (2,))
            angles_deg = np.random.randint(0, 180 + 1, (2,))   # [theta1, theta2]
            structure = LayeredStructure(
                superstrate=superstrate_mat,
                substrate=substrate_mat,
                layers=[
                    MoO3(d=d_layers_nm[0] * 1e-9, phi=np.deg2rad(angles_deg[0])),
                    MoO3(d=d_layers_nm[1] * 1e-9, phi=np.deg2rad(angles_deg[1])),
                ],
            )
            Tss_true = np.array([
                float(calculate_transmission(f, alpha, structure, basis="linear")[1])
                for f in freqs
            ])
            if (np.all(np.isfinite(Tss_true))
                    and Tss_true.max() <= 1.0 and Tss_true.min() >= 0.0):
                break

        print("\nParámetros aleatorios para inferencia:")
        print("d_layers_nm =", d_layers_nm)
        print("angles_deg  =", angles_deg)

        # T_ss de la red (una sola llamada por lotes). Orden: theta1, theta2, d1, d2, freq
        parameters_batch = np.column_stack([
            np.full(len(freqs), angles_deg[0]),
            np.full(len(freqs), angles_deg[1]),
            np.full(len(freqs), d_layers_nm[0]),
            np.full(len(freqs), d_layers_nm[1]),
            freqs,
        ])
        Tss_pred_batch, _ = auxf.predict(
            model, parameters_batch, database, scaler_path=scaler_path
        )
        Tss_pred = np.abs(np.squeeze(Tss_pred_batch))

        lambda_mu = 1e4 / freqs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(lambda_mu, Tss_pred, s=5, c="green", label="T_ss (red)")
        ax.scatter(lambda_mu, Tss_true, s=5, c="red", label="T_ss (TMM)")
        ax.set_xlabel(r"$\lambda$ ($\mu m$)")
        ax.set_ylabel(r"$T_{ss}$ (transmitancia s$\to$s)")
        ax.set_title(
            f"d1={d_layers_nm[0]} nm, d2={d_layers_nm[1]} nm, "
            f"θ1={angles_deg[0]}°, θ2={angles_deg[1]}°"
        )
        ax.legend()
        plt.savefig(
            results_dir
            / f"comparison_d{d_layers_nm[0]}_{d_layers_nm[1]}"
            f"_th{angles_deg[0]}_{angles_deg[1]}.png"
        )
        plt.close(fig)

    print(f"\nTiempo total: {time.time() - t1:.2f} s")


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    run_inference(
        model_dir=str(BASE / "NN_Code" / "Tss_Models_Trained_bilayers_MoO3"),
        database=str(BASE / "NN_Code" / "Dataset_Tss_Bilayer"),
        num_seed=1,
        N_pred=3,
        superstrate_mat=Air(),
        substrate_mat=BaF2(),
        alpha=np.deg2rad(40),
    )
