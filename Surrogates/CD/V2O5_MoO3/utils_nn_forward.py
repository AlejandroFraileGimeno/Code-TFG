import json
import numpy as np

FREQ_MIN = 600.0
FREQ_MAX = 1100.0


def _load_scalers(scaler_path):
    with open(scaler_path) as f:
        s = json.load(f)
    return s["X"], s["y"]


def _scale_X(X, sx):
    mn = np.array(sx["min"])
    mx = np.array(sx["max"])
    return (X - mn) / (mx - mn + 1e-10)


def _unscale_y(y_scaled, sy):
    mn = sy["min"]
    mx = sy["max"]
    return y_scaled * (mx - mn + 1e-10) + mn


def predict(model, params, database, scaler_path=None):
    if scaler_path is None:
        import os
        scaler_path = os.path.join(database, "scalers.json")
    sx, sy = _load_scalers(scaler_path)
    X_scaled = _scale_X(params, sx)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = _unscale_y(y_scaled, sy)
    return y, None