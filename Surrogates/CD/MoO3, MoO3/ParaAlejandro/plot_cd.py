"""
Pinta el espectro de dicroísmo circular (CD) PREDICHO POR EL MODELO.

Al ejecutarlo te PREGUNTA los parámetros por teclado (d1, d2, theta).
Pulsa Enter sin escribir nada para usar el valor por defecto que aparece
entre corchetes.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

import utils_nn_forward as auxf

FREQ_MIN, FREQ_MAX = 600, 1100   # rango de frecuencias (cm-1)
N_FREQS = 1000


def pedir_numero(texto, mini, maxi, defecto):
    """Pide un número por teclado, con valor por defecto y comprobación de rango."""
    while True:
        resp = input(f"{texto} [{mini}-{maxi}, por defecto {defecto}]: ").strip()
        if resp == "":
            return defecto
        try:
            valor = float(resp)
        except ValueError:
            print("  -> Escribe un número válido.")
            continue
        if not (mini <= valor <= maxi):
            print(f"  -> Debe estar entre {mini} y {maxi}.")
            continue
        return valor


def pedir_si_no(texto, defecto=True):
    d = "S" if defecto else "n"
    resp = input(f"{texto} [s/n, por defecto {d}]: ").strip().lower()
    if resp == "":
        return defecto
    return resp.startswith("s")


print("=== Espectro CD del modelo ===")
d1 = pedir_numero("Grosor capa 1 d1 (nm)", 200, 2000, 904)
d2 = pedir_numero("Grosor capa 2 d2 (nm)", 200, 2000, 1711)
theta = pedir_numero("Ángulo entre capas theta (grados)", 0, 180, 60)
comparar = pedir_si_no("¿Comparar con el cálculo físico exacto (TMM)?", True)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "NN_Code" / "Forward_Models_Trained_bilayers_MoO3" / "Model_1seed"

# --- Cargar el modelo entrenado (Keras 3) ---
print("\nCargando modelo...")
model = models.load_model(str(MODEL_DIR / "Model_1seed.h5"), compile=False)
scaler_path = MODEL_DIR / "scalers.json"

# --- Predecir el CD con el modelo para cada frecuencia ---
freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
params = np.column_stack([
    np.full(N_FREQS, theta),
    np.full(N_FREQS, d1),
    np.full(N_FREQS, d2),
    freqs,
])
cd_pred, _ = auxf.predict(model, params, None, scaler_path=scaler_path)
cd_pred = np.abs(np.squeeze(cd_pred))

lambda_mu = 1e4 / freqs  # longitud de onda en micras

# --- Dibujar ---
plt.figure(figsize=(9, 5.5))
plt.plot(lambda_mu, cd_pred, color="#185FA5", lw=2, label="CD (red neuronal)")

if comparar:
    from generalized_transfer_matrix_method import (
        Air, Au, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
    )
    structure = LayeredStructure(
        superstrate=Air(), substrate=Au(),
        layers=[MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)), MoO3(d=d2 * 1e-9)],
    )
    cd_true = np.array([abs(calculate_circular_dichroism_ref(f, 0, structure)[1])
                        for f in freqs])
    plt.plot(lambda_mu, cd_true, color="#D85A30", lw=1.5, ls="--",
             label="CD (TMM, físico exacto)")

plt.xlabel(r"Longitud de onda $\lambda$ ($\mu$m)")
plt.ylabel("|CD| reflexión")
plt.title(f"Espectro CD  ·  d1={d1:.0f} nm, d2={d2:.0f} nm, θ={theta:.0f}°")
plt.legend()
plt.grid(True, alpha=0.3)

out = MODEL_DIR.parent / f"cd_d{d1:.0f}_{d2:.0f}_theta{theta:.0f}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Gráfica guardada en: {out}")
plt.show()
