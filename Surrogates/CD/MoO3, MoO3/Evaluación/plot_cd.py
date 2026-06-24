"""
===========================================================
Plot CD spectrum for a user-defined configuration
===========================================================
Edit d1, d2 and theta below and run.

Author: [Lucia F. Alvarez-Tomillo / Alejandro Fraile]
Date: [xx/xx/2026]
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
    Air, Au, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf

# --- Configuración ---
d1    = 500   # nm
d2    = 800   # nm
theta = 45    # grados
num_seed = 1
alpha    = 0
# ---------------------

freqs      = np.linspace(600, 1100, 1000)
model_dir  = ROOT_PATH / "Models" / "CD" / "MoO3_MoO3"
model_path = model_dir / f"Model_{num_seed}seed" / f"Model_{num_seed}seed.h5"
scaler_path = model_dir / f"Model_{num_seed}seed" / "scalers.json"
database   = str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MoO3")

model = models.load_model(model_path, compile=False)

structure = LayeredStructure(
    superstrate=Air(),
    substrate=Au(),
    layers=[
        MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
        MoO3(d=d2 * 1e-9),
    ],
)

CD_true = np.array([
    abs(calculate_circular_dichroism_ref(f, alpha, structure)[1]) for f in freqs
])

params = np.column_stack([
    np.full(len(freqs), theta),
    np.full(len(freqs), d1),
    np.full(len(freqs), d2),
    freqs,
])
CD_pred_batch, _ = auxf.predict(model, params, database, scaler_path=scaler_path)
CD_pred = np.array([abs(float(np.squeeze(cd))) for cd in CD_pred_batch])

lambda_mu = 1e4 / freqs
plt.figure(figsize=(9, 6))
plt.scatter(lambda_mu, CD_pred, s=5, c="blue", label="CD NN")
plt.scatter(lambda_mu, CD_true, s=5, c="red",  label="CD true")
plt.xlabel(r"$\lambda$ ($\mu$m)")
plt.ylabel("CD reflection")
plt.title(f"d1={d1} nm, d2={d2} nm, θ={theta}°")
plt.legend()
plt.tight_layout()
plt.show()