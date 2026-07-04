# -*- coding: utf-8 -*-
"""
Generación PARALELA de dataset para red tandem — T_xx  V2O5/V2O5/BaF2
=======================================================================
Idéntico a generate_dataset.py pero usa ProcessPoolExecutor.
Divide N_SAMPLES en chunks y procesa cada chunk en un worker independiente.
Combina los chunks al final y guarda los mismos ficheros CSV.

Checkpointing: si un chunk_XXXX.npz ya existe en la carpeta tmp/, lo salta.
"""

import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ============================================================
# CONFIG
# ============================================================
N_SAMPLES  = 500000
D_MIN      = 200
D_MAX      = 1200
FREQ_MIN   = 400.0
FREQ_MAX   = 1400.0
N_FREQS    = 1000
SEED       = 42
N_WORKERS  = 12
CHUNK_SIZE = 500     # muestras por tarea
# ============================================================

ROOT_PATH  = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT_PATH / "Datasets" / "T_xx" / "V2O5_V2O5_BaF2"
TMP_DIR    = OUTPUT_DIR / "tmp_chunks"


def _compute_chunk(chunk_idx: int, theta1s, theta2s, d1s, d2s, freqs):
    """Calcula T_xx para un bloque de muestras. Se ejecuta en worker."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    sys.path.insert(0, str(ROOT_PATH / "TMM"))

    from generalized_transfer_matrix_method import (
        Air, BaF2, V2O5, LayeredStructure, calculate_transmission,
    )

    n   = len(theta1s)
    out = np.empty((n, len(freqs)), dtype=np.float32)

    for i in range(n):
        phi1 = np.deg2rad(theta1s[i])
        phi2 = np.deg2rad(theta2s[i])
        structure = LayeredStructure(
            superstrate=Air(),
            substrate=BaF2(),
            layers=[
                V2O5(d=d1s[i] * 1e-9, phi=phi1),
                V2O5(d=d2s[i] * 1e-9, phi=phi2),
            ],
        )
        for j, f in enumerate(freqs):
            t = calculate_transmission(f, 0, structure, basis="linear")
            out[i, j] = float(t[0])

    return chunk_idx, out


def _run_chunk(args):
    chunk_idx, theta1s, theta2s, d1s, d2s, freqs = args
    out_file = TMP_DIR / f"chunk_{chunk_idx:05d}.npz"
    if out_file.exists():
        return chunk_idx, None   # ya existe

    _, T_xx = _compute_chunk(chunk_idx, theta1s, theta2s, d1s, d2s, freqs)
    np.savez_compressed(out_file, T_xx=T_xx)
    return chunk_idx, T_xx.shape[0]


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    FREQS = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

    np.random.seed(SEED)
    theta1_arr = np.random.uniform(0,     180,   N_SAMPLES)
    theta2_arr = np.random.uniform(0,     180,   N_SAMPLES)
    d1_arr     = np.random.uniform(D_MIN, D_MAX, N_SAMPLES)
    d2_arr     = np.random.uniform(D_MIN, D_MAX, N_SAMPLES)

    # Dividir en chunks
    indices = np.arange(N_SAMPLES)
    chunks  = [indices[i:i + CHUNK_SIZE] for i in range(0, N_SAMPLES, CHUNK_SIZE)]
    n_chunks = len(chunks)

    pending = [
        (ci, theta1_arr[ch], theta2_arr[ch], d1_arr[ch], d2_arr[ch], FREQS)
        for ci, ch in enumerate(chunks)
        if not (TMP_DIR / f"chunk_{ci:05d}.npz").exists()
    ]
    already_done = n_chunks - len(pending)

    print(f"Generación paralela T_xx — V2O5/V2O5/BaF2")
    print(f"N_SAMPLES={N_SAMPLES}  CHUNK_SIZE={CHUNK_SIZE}  N_WORKERS={N_WORKERS}")
    print(f"Chunks totales: {n_chunks}  |  Pendientes: {len(pending)}  |  Ya hechos: {already_done}")
    print(f"Salida: {OUTPUT_DIR}\n")

    t_start   = time.time()
    completed = already_done

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_run_chunk, args): args[0] for args in pending}
        for future in as_completed(futures):
            chunk_idx, n_done = future.result()
            completed += 1
            elapsed = time.time() - t_start
            eta     = elapsed / max(1, completed - already_done) * (n_chunks - completed)
            if n_done is None:
                print(f"  [{completed:4d}/{n_chunks}]  chunk {chunk_idx:05d} -> ya existia")
            else:
                print(f"  [{completed:4d}/{n_chunks}]  chunk {chunk_idx:05d}  "
                      f"({n_done} muestras)  ETA {eta/60:.1f} min", flush=True)

    print(f"\nCombinando {n_chunks} chunks...")
    T_xx_all = np.empty((N_SAMPLES, N_FREQS), dtype=np.float32)
    params    = np.column_stack([theta1_arr, theta2_arr, d1_arr, d2_arr])

    for ci, ch in enumerate(chunks):
        data = np.load(TMP_DIR / f"chunk_{ci:05d}.npz")
        T_xx_all[ch] = data["T_xx"]

    print(f"Guardando en {OUTPUT_DIR} ...")
    np.savetxt(OUTPUT_DIR / "params.csv",       params,    delimiter=",",
               header="theta1_deg,theta2_deg,d1_nm,d2_nm", comments="")
    np.savetxt(OUTPUT_DIR / "T_xx_spectra.csv", T_xx_all,  delimiter=",")
    np.savetxt(OUTPUT_DIR / "freqs.csv",        FREQS,     delimiter=",")

    elapsed_total = time.time() - t_start
    print(f"\nListo. {N_SAMPLES} muestras en {elapsed_total/60:.1f} min")
    print(f"  params.csv       {params.shape}")
    print(f"  T_xx_spectra.csv {T_xx_all.shape}")
    print(f"\nPuedes borrar {TMP_DIR} una vez verificado el dataset.")