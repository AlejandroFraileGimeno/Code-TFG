"""
Plot R_total spectrum for a user-defined configuration (MoO3 / V2O5 bilayer).
Edit d1, d2 and theta below and run.
"""

import sys
from pathlib import Path
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, V2O5, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf

# --- Configuracion ---
d1       = 500   # nm
d2       = 800   # nm
theta    = 45    # grados
num_seed = 1
alpha    = 0
# ---------------------

freqs       = np.linspace(600, 1100, 1000)
model_dir   = ROOT_PATH / "Models" / "R_total" / "MoO3_V2O5"
model_path  = model_dir / f"Model_{num_seed}seed" / f"Model_{num_seed}seed.h5"
scaler_path = model_dir / f"Model_{num_seed}seed" / "scalers.json"
database    = str(ROOT_PATH / "Datasets" / "CD" / "MoO3_V2O5")

model = models.load_model(model_path, compile=False)

structure = LayeredStructure(
    superstrate=Air(), substrate=Au(),
    layers=[
        MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
        V2O5(d=d2 * 1e-9),
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
plt.figure(figsize=(9, 6))
plt.scatter(lambda_mu, R_pred, s=5, c="blue", label="R_total NN")
plt.scatter(lambda_mu, R_true, s=5, c="red",  label="R_total TMM")
plt.xlabel(r"$\lambda$ ($\mu$m)")
plt.ylabel(r"$R_{total} = R_r + R_l$")
plt.title(f"MoO3/V2O5 — d1={d1} nm, d2={d2} nm, theta={theta} deg")
plt.legend()
plt.tight_layout()
plt.show()