# -*- coding: utf-8 -*-
"""
Generación de dataset para red tandem — T_xx  MoO3/V2O5/BaF2
==============================================================
Estructura: Air / MoO3(d1, phi=theta1) / V2O5(d2, phi=theta2) / BaF2

Parámetros muestreados aleatoriamente:
  theta1 : [0, 180]  grados  (rotación MoO3)
  theta2 : [0, 180]  grados  (rotación V2O5)
  d1     : [D_MIN, D_MAX] nm
  d2     : [D_MIN, D_MAX] nm

Salidas por muestra:
  T_xx(f) = T_pp de calculate_transmission(basis="linear")  (N_FREQS valores)

Ficheros generados en Datasets/T_xx/MoO3_V2O5_BaF2/:
  params.csv        (N, 4)        theta1, theta2, d1, d2
  T_xx_spectra.csv  (N, N_FREQS)  espectro T_xx
  freqs.csv         (N_FREQS,)    eje de frecuencias en cm-1
"""

import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# CONFIG
# ============================================================
N_SAMPLES = 50_000
D_MIN     = 200      # nm
D_MAX     = 2000     # nm
FREQ_MIN  = 600.0    # cm-1
FREQ_MAX  = 1000.0   # cm-1
N_FREQS   = 1000
SEED      = 42
# ============================================================

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, V2O5, LayeredStructure,
    calculate_transmission,
)

OUTPUT_DIR = ROOT_PATH / "Datasets" / "T_xx" / "MoO3_V2O5_BaF2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQS = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

np.random.seed(SEED)
theta1_arr = np.random.uniform(0,     180,   N_SAMPLES)
theta2_arr = np.random.uniform(0,     180,   N_SAMPLES)
d1_arr     = np.random.uniform(D_MIN, D_MAX, N_SAMPLES)
d2_arr     = np.random.uniform(D_MIN, D_MAX, N_SAMPLES)

params       = np.column_stack([theta1_arr, theta2_arr, d1_arr, d2_arr])
T_xx_spectra = np.empty((N_SAMPLES, N_FREQS), dtype=np.float32)

print(f"Generando {N_SAMPLES} espectros T_xx  ({N_FREQS} frecuencias por muestra)...")
print(f"Rango: {FREQ_MIN:.0f}–{FREQ_MAX:.0f} cm-1   d: {D_MIN}–{D_MAX} nm")
print(f"Salida: {OUTPUT_DIR}\n")

t_start = time.time()

for i in range(N_SAMPLES):
    phi1 = np.deg2rad(theta1_arr[i])
    phi2 = np.deg2rad(theta2_arr[i])
    d1   = d1_arr[i] * 1e-9
    d2   = d2_arr[i] * 1e-9

    structure = LayeredStructure(
        superstrate=Air(),
        substrate=BaF2(),
        layers=[
            MoO3(d=d1, phi=phi1),
            V2O5(d=d2, phi=phi2),
        ],
    )

    for j, f in enumerate(FREQS):
        t = calculate_transmission(f, 0, structure, basis="linear")
        T_xx_spectra[i, j] = float(t[0])   # T_pp = T_xx

    if (i + 1) % 1000 == 0 or (i + 1) == N_SAMPLES:
        elapsed = time.time() - t_start
        eta     = elapsed / (i + 1) * (N_SAMPLES - i - 1)
        print(f"  [{i+1:6d}/{N_SAMPLES}]  {elapsed/60:.1f} min  ETA {eta/60:.1f} min",
              flush=True)

print(f"\nGuardando en {OUTPUT_DIR} ...")
np.savetxt(OUTPUT_DIR / "params.csv",       params,       delimiter=",",
           header="theta1_deg,theta2_deg,d1_nm,d2_nm", comments="")
np.savetxt(OUTPUT_DIR / "T_xx_spectra.csv", T_xx_spectra, delimiter=",")
np.savetxt(OUTPUT_DIR / "freqs.csv",        FREQS,        delimiter=",")

elapsed_total = time.time() - t_start
print(f"Listo. {N_SAMPLES} muestras en {elapsed_total/60:.1f} min")
print(f"  params.csv       {params.shape}")
print(f"  T_xx_spectra.csv {T_xx_spectra.shape}")