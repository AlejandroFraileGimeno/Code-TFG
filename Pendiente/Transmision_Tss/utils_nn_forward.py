# -*- coding: utf-8 -*-
"""
===========================================================
Aux_Functions_Rp
===========================================================
This script contains auxiliary functions used in the document.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

# print(tf.__version__)
import os
import sys
from functools import partial
import warnings
import csv
import json

from generalized_transfer_matrix_method import (
    Air,
    SiO2,
    MoO3,
    MgTeMoO6,
    LayeredStructure,
    calculate_circular_dichroism_ref,
)

FREQ_MIN = 600.0
FREQ_MAX = 1100.0


def guardar_datos_en_csv(datos, nombre_archivo):
    """
    Save a list of rows to a CSV file.

    Parameters
    ----------
    datos : list of list
        Data to be saved, each sublist is a row.
    nombre_archivo : str
        Path to the output CSV file.
    """
    with open(nombre_archivo, mode="w", newline="") as archivo:
        escritor_csv = csv.writer(archivo, quoting=csv.QUOTE_ALL)
        for fila in datos:
            escritor_csv.writerow(fila)


# Custom warning class
class MiAdvertencia(Warning):
    pass


# Custom warning renderer
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Custom warning display function that prints warnings in red.
    """
    red = "\033[91m"
    reset = "\033[0m"
    sys.stderr.write(f"{red}{message}{reset}\n")


warnings.showwarning = custom_showwarning


# =============================================================================
# Normalization and data base
# =============================================================================
def read_single_csv(file: str) -> np.ndarray:
    """
    Read a CSV file without header and return as ndarray.
    """
    p = Path(file)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file}")
    X = pd.read_csv(p, header=None, dtype=float)
    return X.to_numpy()


def normalize_data(data, min_val, max_val):
    """
    Normalize data to [0, 1] range.
    """
    if np.any(np.asarray(max_val) == np.asarray(min_val)):
        raise ValueError("Cannot normalize data with zero min/max range.")
    return (data - min_val) / (max_val - min_val)


def unnormalize_data(data_norm, min_val, max_val):
    """
    Undo normalization to original scale.
    """
    return data_norm * (max_val - min_val) + min_val


def standardize_data(data, mean_val, std_val):
    """
    Standardize data to zero mean and unit variance.
    """
    if np.any(np.asarray(std_val) == 0):
        raise ValueError("Cannot standardize data with zero standard deviation.")
    return (data - mean_val) / std_val


def unstandardize_data(data_std, mean_val, std_val):
    """
    Undo standardization to original scale.
    """
    return data_std * std_val + mean_val


def load_database(database):  # normalize_database
    file_angles = database + "/angles.csv"
    file_thickness = database + "/thickness.csv"
    file_CD = database + "/Tss_spectra_norm.csv"  # espectro objetivo (T_ss)
    # Check if the first column of angles.csv is all zeros (bilayer)
    angles_dataset_ref = read_single_csv(file_angles)
    thickness_dataset_ref = read_single_csv(file_thickness)
    CD_dataset_ref = read_single_csv(file_CD)

    return (
        angles_dataset_ref,
        thickness_dataset_ref,
        CD_dataset_ref,
    )


def expand_dataset(angles, thickness, CD, freqs):

    n_structures = CD.shape[0]
    n_freqs = CD.shape[1]

    theta1_column = np.repeat(angles[:, 0], n_freqs)
    theta2_column = np.repeat(angles[:, 1], n_freqs)

    t1_column = np.repeat(thickness[:, 0], n_freqs)
    t2_column = np.repeat(thickness[:, 1], n_freqs)

    freq_column = np.tile(freqs, n_structures)

    cd_column = CD.reshape(-1)

    # Orden de las 5 entradas: theta1, theta2, d1, d2, frecuencia
    X = np.column_stack(
        [theta1_column, theta2_column, t1_column, t2_column, freq_column]
    )

    Y = cd_column.reshape(-1, 1)

    return X, Y


def prepare_data(ntrain, nvalidation, database, return_scalers=False):
    """
    Prepare normalized training and validation data.

    Parameters
    ----------
    ntrain : int
        Number of training samples.
    nvalidation : int
        Number of validation samples.
    database : str
        Path to the dataset folder.

    Returns
    -------
    xtr : dict
        Training input dictionary.
    xva : dict
        Validation input dictionary.
    ytr_single : np.ndarray
        Training targets.
    yva_single : np.ndarray
        Validation targets.
    """
    angles, thickness, CD = load_database(database)
    n_structures = CD.shape[0]
    n_freqs = CD.shape[1]

    freqs = np.linspace(FREQ_MIN, FREQ_MAX, n_freqs)

    ndata_t = ntrain + nvalidation

    # ndata = n_structures * n_freqs  # Total number of samples in the dataset

    # if ndata > ndata_t:
    #     warnings.warn("You are not using the entire database", MiAdvertencia)
    # elif ndata < ndata_t:
    #     raise ValueError(
    #         "Error: The database is smaller than the sum of the selected training and validation values."
    #     )
    # else:
    #     print(
    #         "The database matches the sum of examples used in validation and training."
    #     )

    if ndata_t > n_structures:
        raise ValueError(
            "The database is smaller than ntrain + nvalidation: "
            f"{n_structures} < {ndata_t}."
        )

    min_CD, max_CD = np.amin(CD), np.amax(CD)
    min_angles, max_angles = np.amin(angles), np.amax(angles)
    min_thickness, max_thickness = np.amin(thickness), np.amax(thickness)
    min_frequency, max_frequency = np.amin(freqs), np.amax(freqs)

    # 5 entradas: theta1, theta2, d1, d2, frecuencia
    feature_min = np.array(
        [min_angles, min_angles, min_thickness, min_thickness, min_frequency],
        dtype=float,
    )
    feature_max = np.array(
        [max_angles, max_angles, max_thickness, max_thickness, max_frequency],
        dtype=float,
    )

    angles_norm = normalize_data(angles, min_angles, max_angles)[0:ndata_t, :]
    thickness_norm = normalize_data(thickness, min_thickness, max_thickness)[
        0:ndata_t, :
    ]
    CD_norm = normalize_data(CD, min_CD, max_CD)[0:ndata_t, :]
    frequency_norm = normalize_data(freqs, min_frequency, max_frequency)

    # CD_norm = np.reshape(CD_norm, (ndata_t, n_freqs, 1))

    Xtr, Ytr = expand_dataset(
        angles_norm[0:ntrain],
        thickness_norm[0:ntrain],
        CD_norm[0:ntrain],
        frequency_norm,
    )

    Xva, Yva = expand_dataset(
        angles_norm[ntrain : ntrain + nvalidation],
        thickness_norm[ntrain : ntrain + nvalidation],
        CD_norm[ntrain : ntrain + nvalidation],
        frequency_norm,
    )

    y_mean = np.mean(Ytr)
    y_std = np.std(Ytr)
    Ytr = standardize_data(Ytr, y_mean, y_std)
    Yva = standardize_data(Yva, y_mean, y_std)

    perm_tr = np.random.permutation(len(Xtr))
    Xtr = Xtr[perm_tr]
    Ytr = Ytr[perm_tr]

    perm_va = np.random.permutation(len(Xva))
    Xva = Xva[perm_va]
    Yva = Yva[perm_va]

    xtr = {"input0": Xtr}
    xva = {"input0": Xva}

    scalers = {
        "feature_min": feature_min,
        "feature_max": feature_max,
        "cd_min": float(min_CD),
        "cd_max": float(max_CD),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "n_freqs": int(n_freqs),
    }

    print("Training samples:", Xtr.shape, "Validation samples:", Xva.shape)
    print("Target mean/std after min-max normalization:", y_mean, y_std)

    # Always return 5 values for consistency
    if return_scalers:
        return xtr, xva, Ytr, Yva, scalers
    else:
        return xtr, xva, Ytr, Yva, None


def save_scalers(scalers, path_scalers):
    """
    Save normalization metadata used during training.
    """
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in scalers.items()
    }
    Path(path_scalers).write_text(json.dumps(serializable, indent=2))


def load_scalers(path_scalers):
    """
    Load normalization metadata used during training.
    """
    scalers = json.loads(Path(path_scalers).read_text())
    scalers["feature_min"] = np.asarray(scalers["feature_min"], dtype=float)
    scalers["feature_max"] = np.asarray(scalers["feature_max"], dtype=float)
    return scalers


# =============================================================================
# Network training functions
# =============================================================================


def load_seed_list(seed_file: str = None):
    if seed_file is None:
        seed_file = str(Path(__file__).parent / "SEED_LIST.csv")
    seed_path = Path(seed_file)
    seed_list = pd.read_csv(seed_path, header=None)
    seed_list = np.array(seed_list)
    max_num = np.shape(seed_list)[0]
    seed_list = np.reshape(seed_list, (max_num,))
    return seed_list


def reset_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(int(seed))


def save_history(history_keras, path_hist):
    history_dict = history_keras.history
    training_cost = history_dict["loss"]  # Store training metric values for each epoch
    evaluation_cost = history_dict["val_loss"]

    epochs = len(evaluation_cost)  # Total number of epochs recorded
    xx = np.linspace(0, epochs - 1, epochs)  # Epoch index array [0, ..., epochs-1]

    f1 = open(path_hist, "w")
    for i in range(0, epochs):  # Serialize metrics row-by-row as text
        summary = (
            str(xx[i])
            + " "
            + str(evaluation_cost[i])
            + " "
            + " "
            + str(training_cost[i])
            + " "
            + ""
            + "\n"
        )
        f1.write(summary)

    f1.close()


def save_metrics_classifier(history_keras, path_hist):
    history_dict = history_keras.history
    training_cost = history_dict[
        "categorical_accuracy"
    ]  # Store training metric values for each epoch
    evaluation_cost = history_dict["val_categorical_accuracy"]

    epochs = len(evaluation_cost)  # Total number of epochs recorded
    xx = np.linspace(0, epochs - 1, epochs)  # Epoch index array [0, ..., epochs-1]

    f1 = open(path_hist, "w")
    for i in range(0, epochs):  # Serialize metrics row-by-row as text
        summary = (
            str(xx[i])
            + " "
            + str(evaluation_cost[i])
            + " "
            + " "
            + str(training_cost[i])
            + " "
            + ""
            + "\n"
        )
        f1.write(summary)

    f1.close()


# =============================================================================
# Post-processing predictions
# =============================================================================


def _get_normalization_arrays(database, bilayer):
    """
    Helper to compute all normalization arrays and min/max for q and features.

    Args:
        database (str): Path to the dataset folder.
        bilayer (bool): True if bilayer ordering is used; False for trilayer.

    Returns:
        dict: Contains min_q, max_q, min_arr_input2, max_arr_input2, min_arr_pred_br, max_arr_pred_br.
    """
    angles, thickness, CD = load_database(database)
    min_CD, max_CD = np.amin(CD), np.amax(CD)
    min_angles, max_angles = np.amin(angles), np.amax(angles)
    min_thickness, max_thickness = np.amin(thickness), np.amax(thickness)
    freqs = np.linspace(FREQ_MIN, FREQ_MAX, CD.shape[1])
    min_freq, max_freq = np.amin(freqs), np.amax(freqs)

    min_complete = []
    max_complete = []
    for i in range(len(angles[1])):
        min_complete.append(min_angles)
        max_complete.append(max_angles)
    for j in range(len(thickness[1])):
        min_complete.append(min_thickness)
        max_complete.append(max_thickness)
    min_complete.append(min_freq)
    max_complete.append(max_freq)

    min_arr_pred_single = np.array(min_complete, dtype=float)
    max_arr_pred_single = np.array(max_complete, dtype=float)

    return {
        "min_CD": min_CD,
        "max_CD": max_CD,
        "min_arr_pred_br": min_arr_pred_single,
        "max_arr_pred_br": max_arr_pred_single,
    }


def _normalize_inputs(parameters, norm_arrays):
    """
    Helper to normalize q_real and input2 using normalization arrays.

    Args:
        q_real (np.ndarray): 1D array of q samples (length N).
        norm_arrays (dict): Output of _get_normalization_arrays.

    Returns:
        tuple: (q_real_norm, input2_norm)
    """
    parameters = np.asarray(parameters, dtype=float)
    parameters = np.atleast_2d(parameters)
    parameters_norm = normalize_data(
        parameters, norm_arrays["min_arr_pred_br"], norm_arrays["max_arr_pred_br"]
    )
    return parameters_norm


def predict(model, parameters, database, scaler_path=None, bilayer=True):
    """
    Predict outputs given only q_real (no input2), using shared normalization logic.
    """
    if scaler_path is not None and Path(scaler_path).exists():
        scalers = load_scalers(scaler_path)
        parameters = np.asarray(parameters, dtype=float)
        parameters_norm = normalize_data(
            np.atleast_2d(parameters),
            scalers["feature_min"],
            scalers["feature_max"],
        )
        CD_pred_std = model.predict(parameters_norm, verbose=0)
        CD_pred_norm = unstandardize_data(
            CD_pred_std,
            scalers["y_mean"],
            scalers["y_std"],
        )
        CD_pred = unnormalize_data(
            CD_pred_norm,
            scalers["cd_min"],
            scalers["cd_max"],
        )
        return CD_pred, CD_pred_norm

    warnings.warn(
        "Scaler file not found. Falling back to legacy min-max inference; "
        "retrain the forward model to avoid mean-collapsed predictions.",
        MiAdvertencia,
    )
    norm_arrays = _get_normalization_arrays(database, bilayer)
    parameters_norm = _normalize_inputs(parameters, norm_arrays)
    CD_pred_norm = model.predict(parameters_norm, verbose=0)
    CD_pred = unnormalize_data(
        CD_pred_norm,
        norm_arrays["min_CD"],
        norm_arrays["max_CD"],
    )

    return CD_pred, CD_pred_norm
