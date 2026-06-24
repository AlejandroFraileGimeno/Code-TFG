# -*- coding: utf-8 -*-
"""
===========================================================
Models
===========================================================
Deep Learning models for predicting parameters in twisted multilayer optical systems.
Defines Keras/TensorFlow models for regression tasks using spectral (CD) features.

Model architecture:
- Input 0: 1D array of q values (shape: [CD_norm, 1]) processed through Conv1D and pooling layers.
    - q_size: Number of q points (1000) for the isofrequency contour.

- Latent layer: Dense layer combining spectral and scalar features.
- hidden_neurons_branch: Number of neurons in the hidden layer of each output branch.
- Nfeatures: Number of parameters to predict per branch.
- Nbranches: Number of output branches (parallel predictions).

- Output branches: Multiple parallel branches (Nbranches) each with a hidden layer and an output layer predicting Nfeatures parameters.
The model supports two inputs: a 1D q array and an optional vector of scalar features.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

########## IMPORT PACKAGES ##########

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

########## KERAS MODELS ##########


def model_twistoptics(Parameters):
    """
    Build a Keras model for twisted multilayer optics regression.

    Parameters
    ----------
    Parameters : int
        Number of parameters (thickness, angles, frequency): (input0 shape).
    hidden_neurons_branch : int
        Number of neurons in the hidden layer of each output branch.

    Output: CD at a single frequency (scalar output).
    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model.
    """
    # Input 0: parameters
    input_shape = (Parameters,)
    print("El shape de input0 es:", input_shape)
    input0 = Input(shape=input_shape, name="input0")

    x = Dense(128, activation="relu", kernel_initializer="he_normal")(input0)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=input0, outputs=output)
    model.summary()
    return model
