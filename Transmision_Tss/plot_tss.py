"""
Pinta el espectro de transmitancia T_ss = |t_ss|^2 PREDICHO POR EL MODELO.

Al ejecutarlo te PREGUNTA los parámetros por teclado (d1, d2, theta).
Pulsa Enter sin escribir nada para usar el valor por defecto entre corchetes.
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


print("=== Espectro de transmitancia T_ss del modelo ===")
d1 = pedir_numero("Grosor capa 1 d1 (nm)", 200, 2000, 904)
d2 = pedir_numero("Grosor capa 2 d2 (nm)", 200, 2000, 1711)
theta1 = pedir_numero("Rotación capa 1 theta1 (grados)", 0, 180, 60)
theta2 = pedir_numero("Rotación capa 2 theta2 (grados)", 0, 180, 0)
comparar = pedir_si_no("¿Comparar con el cálculo físico exacto (TMM)?", True)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "NN_Code" / "Tss_Models_Trained_bilayers_MoO3" / "Model_1seed"

# --- Cargar el modelo entrenado (Keras 3) ---
print("\nCargando modelo...")
model = models.load_model(str(MODEL_DIR / "Model_1seed.h5"), compile=False)
scaler_path = MODEL_DIR / "scalers.json"

# --- Predecir T_ss con el modelo para cada frecuencia ---
# Orden de entradas: theta1, theta2, d1, d2, frecuencia
freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
params = np.column_stack([
    np.full(N_FREQS, theta1),
    np.full(N_FREQS, theta2),
    np.full(N_FREQS, d1),
    np.full(N_FREQS, d2),
    freqs,
])
t_pred, _ = auxf.predict(model, params, None, scaler_path=scaler_path)
t_pred = np.abs(np.squeeze(t_pred))

lambda_mu = 1e4 / freqs  # longitud de onda en micras

# --- Dibujar ---
plt.figure(figsize=(9, 5.5))
plt.plot(lambda_mu, t_pred, color="#188A4A", lw=2, label="T_ss (red neuronal)")

if comparar:
    from generalized_transfer_matrix_method import (
        Air, BaF2, MoO3, LayeredStructure, calculate_transmission,
    )
    structure = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta1)),
                MoO3(d=d2 * 1e-9, phi=np.deg2rad(theta2))],
    )
    t_true = np.array([float(calculate_transmission(f, 0, structure, basis="linear")[1])
                       for f in freqs])
    if t_true.max() > 1.0:
        print("AVISO: esta estructura es numericamente inestable en el TMM "
              f"(T_ss llega a {t_true.max():.2e} > 1). Esos puntos son artefactos; "
              "se recortan a 1 solo para la grafica.")
        t_true = np.clip(t_true, 0.0, 1.0)
    plt.plot(lambda_mu, t_true, color="#D85A30", lw=1.5, ls="--",
             label="T_ss (TMM, físico exacto)")

plt.xlabel(r"Longitud de onda $\lambda$ ($\mu$m)")
plt.ylabel(r"$T_{ss}$ (transmitancia s$\to$s)")
plt.title(f"Espectro T_ss  ·  d1={d1:.0f} d2={d2:.0f} nm, θ1={theta1:.0f}° θ2={theta2:.0f}°")
plt.legend()
plt.grid(True, alpha=0.3)

out = MODEL_DIR.parent / f"tss_d{d1:.0f}_{d2:.0f}_th{theta1:.0f}_{theta2:.0f}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Gráfica guardada en: {out}")
plt.show()
