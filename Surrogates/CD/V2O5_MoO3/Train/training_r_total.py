import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers


def train_r_total(database, output_dir, ntrain=8000, nvalidation=2000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    print(f"SEED =  {seed}")

    from forward_model import build_model

    angles    = np.loadtxt(database + "/angles.csv",    delimiter=",")
    thickness = np.loadtxt(database + "/thickness.csv", delimiter=",")
    rt        = np.loadtxt(database + "/R_total_spectra.csv", delimiter=",")

    n_data, n_freq = rt.shape
    theta = angles.flatten()
    d1    = thickness[:, 0]
    d2    = thickness[:, 1]
    freqs = np.linspace(600, 1100, n_freq)

    theta_rep = np.repeat(theta, n_freq)
    d1_rep    = np.repeat(d1,    n_freq)
    d2_rep    = np.repeat(d2,    n_freq)
    freq_rep  = np.tile(freqs,   n_data)
    rt_flat   = rt.flatten()

    X = np.column_stack([theta_rep, d1_rep, d2_rep, freq_rep])
    y = rt_flat

    idx = np.random.permutation(n_data)
    tr_idx  = idx[:ntrain]
    val_idx = idx[ntrain:ntrain + nvalidation]

    def rows(idx_arr):
        return np.concatenate([np.arange(i * n_freq, (i+1) * n_freq) for i in idx_arr])

    X_tr  = X[rows(tr_idx)];  y_tr  = y[rows(tr_idx)]
    X_val = X[rows(val_idx)]; y_val = y[rows(val_idx)]

    X_min = X_tr.min(axis=0); X_max = X_tr.max(axis=0)
    y_min = float(y_tr.min()); y_max = float(y_tr.max())
    X_tr_s  = (X_tr  - X_min) / (X_max - X_min + 1e-10)
    X_val_s = (X_val - X_min) / (X_max - X_min + 1e-10)
    y_tr_s  = (y_tr  - y_min) / (y_max - y_min + 1e-10)
    y_val_s = (y_val - y_min) / (y_max - y_min + 1e-10)

    import os
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + "/scalers.json", "w") as f:
        json.dump({"X": {"min": X_min.tolist(), "max": X_max.tolist()},
                   "y": {"min": y_min, "max": y_max}}, f)

    model = build_model(input_dim=4)
    model.compile(optimizer=optimizers.Adam(1e-3), loss=losses.Huber(delta=1.0))

    model_name = Path(output_dir).name
    cb = [
        callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        callbacks.ModelCheckpoint(output_dir + f"/{model_name}.h5", save_best_only=True),
    ]
    model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
              epochs=200, batch_size=4096, callbacks=cb, verbose=1)
    print(f"---Training completed---")