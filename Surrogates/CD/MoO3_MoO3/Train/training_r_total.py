# -*- coding: utf-8 -*-
"""
===========================================================
Training — R_total surrogate
===========================================================
Igual que training_forward.py pero entrenando sobre R_total = R_r + R_l
en vez de CD_norm. Los modelos se guardan en Models/R_total/MoO3_MoO3/.

Author: [Lucia F. Alvarez-Tomillo / Alejandro Fraile]
Date:   [xx/xx/2026]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import forward_model as mtlm
import utils_nn_forward as auxf


def train_models(
    directory: str = "./Models_Trained_R_total/",
    database: str = "./Database_bilayers_MoO3",
    ntrain: int = 800,
    nvalidation: int = 200,
    resume_training: bool = False,
) -> None:
    directory = str(Path(directory))
    database = str(Path(database))
    Path(directory).mkdir(parents=True, exist_ok=True)

    nfeatures = 4

    xtr, xva, ytr, yva, scalers = auxf.prepare_data(
        ntrain,
        nvalidation,
        database,
        return_scalers=True,
        target_file="R_total_spectra.csv",
    )

    early_stopping = True
    epochs = 200 if early_stopping else 80
    minib_size = 4096

    seed_list = auxf.load_seed_list()
    num_simulations = 5
    list_simulations = [i for i in range(num_simulations)]

    for i in list_simulations:
        print("SEED = ", seed_list[i])
        auxf.reset_random_seeds(seed_list[i + 1])

        model = mtlm.model_twistoptics(nfeatures)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=["mean_squared_error"],
        )

        file_name = f"Model_{i + 1}seed"
        folder = Path(directory) / file_name
        folder.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        r = model.fit(
            xtr,
            ytr,
            batch_size=minib_size,
            epochs=epochs,
            verbose=1,
            validation_data=(xva, yva),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-6,
                    patience=20,
                    mode="min",
                    restore_best_weights=True,
                )
            ]
            if early_stopping
            else [],
        )
        print(f"---Training completed in {time.time() - start_time} seconds ---")

        model.save(folder / f"{file_name}.h5", save_format="h5")
        auxf.save_scalers(scalers, folder / "scalers.json")
        auxf.save_history(r, folder / "history_loss.txt")
        (folder / "hyperparameters.txt").write_text(
            f"n_train={ntrain}\n"
            f"n_val={nvalidation}\n"
            f"epochs={epochs}\n"
            f"mbs={minib_size}\n"
            f"loss=Huber(delta=1.0)\n"
            f"optimizer=Adam(learning_rate=1e-3)\n"
            f"target=standardized_minmax_R_total"
        )