from tensorflow.keras import layers, models


def build_model(input_dim=4):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64,  activation="relu")(x)
    x = layers.Dense(32,  activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    return models.Model(inp, out)