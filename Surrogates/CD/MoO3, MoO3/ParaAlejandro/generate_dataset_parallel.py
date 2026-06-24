# -*- coding: utf-8 -*-
"""
Generador de dataset PARALELO (bicapa MoO3) para la red de Lucia.

Reparte las estructuras entre los nucleos de la CPU. Produce los mismos 3 CSV
que espera utils_nn_forward.load_database:
    angles.csv  /  thickness.csv  /  CD_spectra_norm.csv

Reproducible: semilla fija (numpy default_rng(SEED)). Equivalente fisico a
data_generation.generate_data, pero mucho mas rapido.

Uso:  python generate_dataset_parallel.py
"""

import os
import sys
import time
from pathlib import Path

# Evita que numpy/BLAS lance hilos dentro de cada worker (oversubscription).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_generation import save_dataset

from generalized_transfer_matrix_method import (
    Air,
    Au,
    MoO3,
    LayeredStructure,
    calculate_circular_dichroism_ref,
)

# -------- Configuracion (coincide con generate_database.py) --------
N_DATA = 10000          # nº de estructuras (train.py usa 8000 + 2000)
N_FREQS = 1000
FREQ_MIN, FREQ_MAX = 600.0, 1100.0
D_MIN, D_MAX = 200, 2000
ALPHA = 0
SEED = 0
N_WORKERS = 10
OUTPUT_DIR = Path(__file__).resolve().parent / "NN_Code" / "Dataset_MoO3_Bilayer"

_FREQS = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)


def _compute(params):
    """Calcula el espectro |CD_norm| de una estructura. Ejecuta en cada worker."""
    d1, d2, theta = params
    structure = LayeredStructure(
        superstrate=Air(),
        substrate=Au(),
        layers=[
            MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
            MoO3(d=d2 * 1e-9),
        ],
    )
    out = np.empty(N_FREQS, dtype=float)
    for j in range(N_FREQS):
        out[j] = abs(calculate_circular_dichroism_ref(_FREQS[j], ALPHA, structure)[1])
    return out


def main():
    rng = np.random.default_rng(SEED)
    d = rng.integers(D_MIN, D_MAX + 1, size=(N_DATA, 2))
    theta = rng.integers(0, 180 + 1, size=N_DATA)
    params = [(int(d[i, 0]), int(d[i, 1]), int(theta[i])) for i in range(N_DATA)]

    CD = np.empty((N_DATA, N_FREQS), dtype=float)

    print(f"Generando {N_DATA} estructuras x {N_FREQS} freqs con {N_WORKERS} procesos...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(_compute, params, chunksize=20)):
            CD[i] = res
            if (i + 1) % 500 == 0 or (i + 1) == N_DATA:
                el = time.time() - t0
                eta = el / (i + 1) * (N_DATA - i - 1)
                print(f"  {i + 1}/{N_DATA}  ({el:.0f}s, ETA {eta:.0f}s)", flush=True)

    angles = theta.reshape(-1, 1).astype(float)
    thickness = d.astype(float)
    save_dataset(CD, angles, thickness, OUTPUT_DIR)
    print(f"LISTO en {time.time() - t0:.0f}s. CD rango: "
          f"[{np.nanmin(CD):.3e}, {np.nanmax(CD):.3e}]")


if __name__ == "__main__":
    main()
