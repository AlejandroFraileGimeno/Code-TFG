"""
Plot R_total spectrum for a user-defined configuration (MoO3 / MgTeMoO6).
Edit d1, d2 and theta below and run.
"""

import sys
from pathlib import Path
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Estilo TFG (solo estética; no afecta a los cálculos)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           12,
    "axes.labelsize":      13,
    "axes.titlesize":      12,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     11,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, MgTeMoO6, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf

# --- Configuracion ---
d1       = 500   # nm  (MoO3)
d2       = 800   # nm  (MgTeMoO6)
theta    = 45    # grados
num_seed = 1
alpha    = 0
# ---------------------

freqs       = np.linspace(600, 1100, 1000)
model_dir   = ROOT_PATH / "Models" / "R_total" / "MoO3_MgTeMoO6"
model_path  = model_dir / f"Model_{num_seed}seed" / f"Model_{num_seed}seed.h5"
scaler_path = model_dir / f"Model_{num_seed}seed" / "scalers.json"
database    = str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MgTeMoO6")

model = models.load_model(model_path, compile=False)

structure = LayeredStructure(
    superstrate=Air(), substrate=Au(),
    layers=[
        MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
        MgTeMoO6(d=d2 * 1e-9),
    ],
)

R_true = np.array([
    calculate_circular_dichroism_ref(f, alpha, structure)[2] for f in freqs
])

params = np.column_stack([
    np.full(len(freqs), theta), np.full(len(freqs), d1),
    np.full(len(freqs), d2), freqs,
])
R_pred_batch, _ = auxf.predict(model, params, database, scaler_path=str(scaler_path))
R_pred = np.array([float(np.squeeze(r)) for r in R_pred_batch])

lambda_mu = 1e4 / freqs
plt.figure(figsize=(7, 4.8))
plt.plot(lambda_mu, R_true, color="#0b0b0b", lw=2.0, label="TMM")
plt.plot(lambda_mu, R_pred, color="#2a78d6", lw=1.6, ls="--", label="NN")
plt.xlabel(r"$\lambda$ ($\mu$m)")
plt.ylabel(r"$R_\mathrm{total} = R_r + R_l$")
plt.title(rf"MoO3/MgTeMoO6 — $d_1$={d1} nm, $d_2$={d2} nm, $\theta$={theta}°")
plt.legend()
plt.tight_layout()
plt.show()