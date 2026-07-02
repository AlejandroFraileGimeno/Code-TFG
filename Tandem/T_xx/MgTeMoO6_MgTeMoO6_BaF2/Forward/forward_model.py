# -*- coding: utf-8 -*-
"""
Red forward espectral para T_xx — MgTeMoO6/MgTeMoO6/BaF2

Entrada : (theta1, theta2, d1, d2)  ->  4 escalares
Salida  : T_xx(f)                   ->  N_FREQS valores en [0, 1]
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization


def build_forward(n_freqs: int) -> Model:
    inp = Input(shape=(4,), name="params")
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(inp)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    out = Dense(n_freqs, activation="sigmoid", name="T_xx")(x)
    model = Model(inputs=inp, outputs=out, name="forward_T_xx")
    model.summary()
    return model