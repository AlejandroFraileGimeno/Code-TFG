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
import time
from functools import partial
from pathlib import Path
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt

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
    num_seed=0,
    N_pred=2,
    superstrate_mat=Air(),
    substrate_mat=Au(),
    alpha=0,
):
    """Load a trained model and produce CD spectra.

    The function reproduces the original prediction pipeline: build CD spectra
    inputs, run multi-branch inference, select best candidates using the oracle
    distance metric, and save comparison plots to ``resultsCanalization``.
    """
    t1 = time.time()
    nfeatures = 4

    model_path = Path(model_dir) / f"Model_{num_seed}seed" / f"Model_{num_seed}seed.h5"
    scaler_path = Path(model_dir) / f"Model_{num_seed}seed" / "scalers.json"
    results_dir = Path(model_dir) / "results"
    os.makedirs(results_dir, exist_ok=True)

    model = models.load_model(
        model_path,
        compile=False,
    )
    print("Modelo:")
    model.summary()
    for i in range(N_pred):
        # SYSTEM PARAMETERS (random CDs)

        freqs = np.linspace(600, 1100, 1000)
        wavelength_mu = 1e4 / freqs  # microns
        d_layers_nm = np.random.randint(
            200, 2000 + 1, (2,)
        )  # d_layers_nm = np.concatenate(([0], d_layers_nm))
        angles_deg = np.random.randint(
            0, 180 + 1, (1,)
        )  # angles_deg = np.concatenate(([0], angles_deg))

        print("\n")
        print("Randomly generated parameters for inference:")
        print("d_layers_nm = ", d_layers_nm)
        print("angles_deg = ", angles_deg)

        structure = LayeredStructure(
            superstrate=superstrate_mat,
            substrate=substrate_mat,
            layers=[
                MoO3(d=d_layers_nm[0] * 1e-9, phi=np.deg2rad(angles_deg[0])),
                MoO3(d=d_layers_nm[1] * 1e-9),
            ],
        )

        # Calculate TMM CD for all frequencies
        CD_list_true = []
        for j in range(len(freqs)):
            CD = calculate_circular_dichroism_ref(freqs[j], alpha, structure)
            CD_list_true.append(abs(CD[1]))

        # Build batch of parameters for all frequencies (single NN call)
        parameters_list = []
        for j in range(len(freqs)):
            parameters = np.concatenate((angles_deg, d_layers_nm, np.array([freqs[j]])))
            parameters_list.append(parameters)
        parameters_batch = np.array(parameters_list)

        # Single batch prediction instead of 1000 individual calls
        CD_pred_batch, _ = auxf.predict(
            model,
            parameters_batch,
            database,
            scaler_path=scaler_path,
        )
        CD_list_pred = [abs(float(np.squeeze(cd))) for cd in CD_pred_batch]

        CD_true = np.array(CD_list_true)
        CD_pred = np.array(CD_list_pred)

        plt.rcParams["figure.figsize"] = (9, 9)
        fig, ax = plt.subplots()
        lambda_nm = 1e4 / freqs
        scatter = ax.scatter(lambda_nm, CD_pred, s=5, c="blue", label="CD NN")
        scatter = ax.scatter(lambda_nm, CD_true, s=5, c="red", label="CD true")
        ax.set_xlabel(r"$\lambda$ ($\mu m$)")
        ax.set_ylabel("CD reflection")
        # ax.set_ylabel("Circular dichroism reflection")

        ax.set_title(
            f"d1={d_layers_nm[0]} nm, d2={d_layers_nm[1]} nm, theta={angles_deg[0]} deg"
        )
        plt.legend()
        plt.savefig(
            results_dir
            / f"comparison_d{d_layers_nm[0]}_{d_layers_nm[1]}_theta{angles_deg[0]}.png"
        )
        plt.close(fig)
        # plt.show()

    print(f"Total execution time: {time.time() - t1:.2f} seconds")


if __name__ == "__main__":
    # Rutas relativas al propio proyecto (funcionan en cualquier equipo).
    BASE = Path(__file__).resolve().parent
    run_inference(
        model_dir=str(BASE / "NN_Code" / "Forward_Models_Trained_bilayers_MoO3"),
        database=str(BASE / "NN_Code" / "Dataset_MoO3_Bilayer"),
        num_seed=1,        # coincide con Model_1seed que genera train.py
        N_pred=3,          # nº de espectros aleatorios a comparar (NN vs TMM)
        superstrate_mat=Air(),
        substrate_mat=Au(),
        alpha=0,
    )
