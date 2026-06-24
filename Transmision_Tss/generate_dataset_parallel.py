# -*- coding: utf-8 -*-
"""
Generador de dataset PARALELO para el proyecto T_ss.

Aprende la TRANSMITANCIA T_ss = |t_ss|^2 (s->s) de una bicapa de MoO3 con AMBAS
capas rotadas (theta1, theta2), sobre substrato de BaF2, a incidencia normal.
Reparte las estructuras entre los nucleos de la CPU.

Produce los 3 CSV que espera utils_nn_forward.load_database:
    angles.csv (2 cols: theta1, theta2) / thickness.csv / Tss_spectra_norm.csv

Reproducible: semilla fija (numpy default_rng(SEED)).

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
    BaF2,
    MoO3,
    LayeredStructure,
    calculate_transmission,
)

# -------- Configuracion (coincide con generate_database.py) --------
N_DATA = 10000          # nº de estructuras (train.py usa 8000 + 2000)
N_FREQS = 1000
FREQ_MIN, FREQ_MAX = 600.0, 1100.0
D_MIN, D_MAX = 200, 2000
ALPHA_DEG = 0           # angulo de incidencia (grados): incidencia normal
ALPHA = np.deg2rad(ALPHA_DEG)
SEED = 0
N_WORKERS = 10
OUTPUT_DIR = Path(__file__).resolve().parent / "NN_Code" / "Dataset_Tss_Bilayer"

_FREQS = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)


def _compute(params):
    """Calcula el espectro T_ss = |t_ss|^2 de una estructura (en cada worker).

    Devuelve None si la estructura NO es fisica (algun T_ss > 1 o no finito):
    el TMM es inestable en capas gruesas/absorbentes y a veces explota. Esas
    estructuras se RECHAZAN y se sustituyen por otras (rejection sampling).
    """
    d1, d2, theta1, theta2 = params
    structure = LayeredStructure(
        superstrate=Air(),
        substrate=BaF2(),   # substrato de BaF2 (transmision)
        layers=[
            MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta1)),
            MoO3(d=d2 * 1e-9, phi=np.deg2rad(theta2)),
        ],
    )
    out = np.empty(N_FREQS, dtype=float)
    for j in range(N_FREQS):
        t = calculate_transmission(_FREQS[j], ALPHA, structure, basis="linear")
        out[j] = float(t[1])  # componente s->s
    # Validacion fisica: transmitancia en [0,1] y finita en TODA la malla.
    if not np.all(np.isfinite(out)) or out.max() > 1.0 or out.min() < 0.0:
        return None
    return out


def _draw_params(rng, n):
    d = rng.integers(D_MIN, D_MAX + 1, size=(n, 2))
    theta = rng.integers(0, 180 + 1, size=(n, 2))   # [theta1, theta2]
    return [(int(d[i, 0]), int(d[i, 1]), int(theta[i, 0]), int(theta[i, 1]))
            for i in range(n)]


def main():
    rng = np.random.default_rng(SEED)

    Tss = np.empty((N_DATA, N_FREQS), dtype=float)
    angles = np.empty((N_DATA, 2), dtype=float)
    thickness = np.empty((N_DATA, 2), dtype=float)

    n_ok = 0
    n_rechazadas = 0
    print(f"Generando {N_DATA} estructuras FISICAS (T_ss<=1) con {N_WORKERS} procesos...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        while n_ok < N_DATA:
            faltan = N_DATA - n_ok
            # Pedimos un ~25% extra para cubrir las que se rechacen (~15%).
            candidatos = _draw_params(rng, int(faltan * 1.25) + 20)
            for p, res in zip(candidatos, ex.map(_compute, candidatos, chunksize=20)):
                if res is None:
                    n_rechazadas += 1
                    continue
                Tss[n_ok] = res
                angles[n_ok] = (p[2], p[3])      # theta1, theta2
                thickness[n_ok] = (p[0], p[1])   # d1, d2
                n_ok += 1
                if n_ok % 500 == 0 or n_ok == N_DATA:
                    el = time.time() - t0
                    eta = el / n_ok * (N_DATA - n_ok)
                    print(f"  {n_ok}/{N_DATA}  (rechazadas {n_rechazadas}, "
                          f"{el:.0f}s, ETA {eta:.0f}s)", flush=True)
                if n_ok == N_DATA:
                    break

    save_dataset(Tss, angles, thickness, OUTPUT_DIR)
    pct = 100 * n_rechazadas / (n_ok + n_rechazadas)
    print(f"LISTO en {time.time() - t0:.0f}s. Rechazadas {n_rechazadas} "
          f"({pct:.1f}%). T_ss rango: [{Tss.min():.3e}, {Tss.max():.3e}]")


if __name__ == "__main__":
    main()
