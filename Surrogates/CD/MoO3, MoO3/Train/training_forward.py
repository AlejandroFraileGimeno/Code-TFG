# -*- coding: utf-8 -*-
"""
===========================================================
Training
===========================================================
Training utilities for multi-branch TwistOptics models.

directory: Required to specify the full path where trained models will be saved
database: Could be replaced by maximum and minimum values from the database
ntrain: Depends on database size
nvalidation: Depends on database size
max_nbranches: Maximum number of branches (default: 10)
min_branch: Minimum number of branches (default: 1)
resume_training: To continue an interrupted training, specify the number of branches from which to resume.
    The model with that number of branches must exist in the specified directory.
resume_nbranches: The branch from which training will resume. The model with that number of branches must exist in the specified directory.

Author: [Lucia F. Alvarez-Tomillo]
Date: [xx/xx/2026]
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import forward_model as mtlm
import utils_nn_forward as auxf


def train_models(
    directory: str = "./Models_Trained_bilayers_MoO3/",
    database: str = "./Database_bilayers_MoO3_MoO3_MoO3",
    ntrain: int = 800,
    nvalidation: int = 200,
    resume_training: bool = False,
) -> None:
    """Train one or more TwistOptics regression models.

    This function preserves the original branch-pruning training workflow used in the
    research code while exposing the main configuration points as function arguments.
    Models, histories, and activity summaries are written under ``directory``.
    """
    directory = str(Path(directory))
    database = str(Path(database))
    Path(directory).mkdir(parents=True, exist_ok=True)

    nfeatures = 4  # Number of parameters of input(e.g., d1, d2, theta, omega)

    xtr, xva, ytr, yva, scalers = auxf.prepare_data(
        ntrain,
        nvalidation,
        database,
        return_scalers=True,
    )

    early_stopping = True
    epochs = 200 if early_stopping else 80
    minib_size = 4096

    seed_list = auxf.load_seed_list()
    num_simulations = 1  # change for training multiple seeds
    list_simulations = [i for i in range(num_simulations)]

    # Loop over seeds
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
            f"target=standardized_minmax_CD"
        )
