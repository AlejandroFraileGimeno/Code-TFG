# -*- coding: utf-8 -*-
"""
Red inversa para diseno inverso de T_xx — MgTeMoO6/MgTeMoO6/BaF2

Entrada : T_xx(f)  ->  espectro objetivo  (N_FREQS valores en [0,1])
Salida  : (theta1, theta2, d1, d2)  normalizado a [0,1]  ->  4 valores
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout


def build_inverse(n_freqs: int) -> Model:
    inp = Input(shape=(n_freqs,), name="T_xx_target")
    x = Dense(1024, activation="relu", kernel_initializer="he_normal")(inp)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    out = Dense(4, activation="sigmoid", name="params_norm")(x)
    model = Model(inputs=inp, outputs=out, name="inverse_T_xx")
    model.summary()
    return model