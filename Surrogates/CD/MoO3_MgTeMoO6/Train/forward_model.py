# -*- coding: utf-8 -*-
"""
===========================================================
Models — MoO3 / MgTeMoO6
===========================================================
Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense


def model_twistoptics(Parameters):
    input_shape = (Parameters,)
    print("El shape de input0 es:", input_shape)
    input0 = Input(shape=input_shape, name="input0")

    x = Dense(128, activation="relu", kernel_initializer="he_normal")(input0)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(64,  activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(32,  activation="relu", kernel_initializer="he_normal")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=input0, outputs=output)
    model.summary()
    return model