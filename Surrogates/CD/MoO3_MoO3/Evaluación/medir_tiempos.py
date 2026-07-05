# -*- coding: utf-8 -*-
"""
===========================================================
medir_tiempos
===========================================================
Compara el tiempo de computo del TMM frente al surrogate (NN)
para el par MoO3/MoO3, a una frecuencia fija, sobre N estructuras
aleatorias (theta, d1, d2) equivalente a "N simulaciones".

- TMM: bucle secuencial de N llamadas a calculate_circular_dichroism_ref
  (no vectorizable, una estructura distinta por iteracion).
- Surrogate: dos modos de medicion, seleccionables con --modo:
    * loop  -> N llamadas a model.predict() de 1 sola fila cada vez
               (mismo patron "una simulacion = una llamada" que el TMM,
               pero penaliza mucho al surrogate por overhead de Keras).
    * batch -> una unica llamada a model.predict() con las N filas
               apiladas (uso vectorizado real del surrogate).
  Ambos usan un unico modelo (Model_1seed).

Con --tipo espectro_completo, cada "simulacion" es el espectro entero
(N_FREQS puntos entre FREQ_MIN y FREQ_MAX, igual que el dataset de
entrenamiento) en vez de un unico punto de frecuencia:
    - TMM: por estructura se construye una vez y se recorre el espectro
      llamando a calculate_circular_dichroism_ref freq a freq.
    - Surrogate: se apilan todas las filas (estructura x frecuencia) y se
      hace una unica llamada batch a predict().

Author: Alejandro Fraile
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from tensorflow.keras import models

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generalized_transfer_matrix_method import (
    Air, Au, MoO3, LayeredStructure, calculate_circular_dichroism_ref,
)
import utils_nn_forward as auxf

FREQ = 850.0        # cm-1, frecuencia fija para las N simulaciones (--tipo frecuencia_unica)
N = 100_000          # numero de simulaciones (estructuras aleatorias)
D_MIN, D_MAX = 200, 2000
ALPHA = 0

FREQ_MIN, FREQ_MAX, N_FREQS = 600.0, 1100.0, 1000  # malla del espectro completo (--tipo espectro_completo)

MODEL_DIR = ROOT_PATH / "Models" / "CD" / "MoO3_MoO3" / "Model_1seed"
DATABASE = str(ROOT_PATH / "Datasets" / "CD" / "MoO3_MoO3")


def generar_estructuras(n, seed=0):
    rng = np.random.default_rng(seed)
    d1 = rng.integers(D_MIN, D_MAX + 1, size=n)
    d2 = rng.integers(D_MIN, D_MAX + 1, size=n)
    theta = rng.integers(0, 181, size=n)
    return d1, d2, theta


def medir_tmm(d1_arr, d2_arr, theta_arr, freq):
    t0 = time.perf_counter()
    for d1, d2, theta in zip(d1_arr, d2_arr, theta_arr):
        structure = LayeredStructure(
            superstrate=Air(),
            substrate=Au(),
            layers=[
                MoO3(d=float(d1) * 1e-9, phi=np.deg2rad(float(theta))),
                MoO3(d=float(d2) * 1e-9),
            ],
        )
        cd = calculate_circular_dichroism_ref(freq, ALPHA, structure)
        _ = abs(cd[1])
    return time.perf_counter() - t0


def medir_surrogate_loop(d1_arr, d2_arr, theta_arr, freq, model, scaler_path):
    t0 = time.perf_counter()
    for d1, d2, theta in zip(d1_arr, d2_arr, theta_arr):
        params = np.array([[theta, d1, d2, freq]], dtype=float)
        _ = auxf.predict(model, params, DATABASE, scaler_path=scaler_path)
    return time.perf_counter() - t0


def medir_surrogate_batch(d1_arr, d2_arr, theta_arr, freq, model, scaler_path):
    params = np.column_stack([
        theta_arr.astype(float),
        d1_arr.astype(float),
        d2_arr.astype(float),
        np.full(len(theta_arr), freq, dtype=float),
    ])
    t0 = time.perf_counter()
    _ = auxf.predict(model, params, DATABASE, scaler_path=scaler_path)
    return time.perf_counter() - t0


def medir_tmm_espectro(d1_arr, d2_arr, theta_arr, freqs):
    t0 = time.perf_counter()
    for d1, d2, theta in zip(d1_arr, d2_arr, theta_arr):
        structure = LayeredStructure(
            superstrate=Air(),
            substrate=Au(),
            layers=[
                MoO3(d=float(d1) * 1e-9, phi=np.deg2rad(float(theta))),
                MoO3(d=float(d2) * 1e-9),
            ],
        )
        for f in freqs:
            cd = calculate_circular_dichroism_ref(f, ALPHA, structure)
            _ = abs(cd[1])
    return time.perf_counter() - t0


def medir_surrogate_batch_espectro(d1_arr, d2_arr, theta_arr, freqs, model, scaler_path):
    n_structures = len(theta_arr)
    theta_col = np.repeat(theta_arr.astype(float), len(freqs))
    d1_col = np.repeat(d1_arr.astype(float), len(freqs))
    d2_col = np.repeat(d2_arr.astype(float), len(freqs))
    freq_col = np.tile(freqs, n_structures)
    params = np.column_stack([theta_col, d1_col, d2_col, freq_col])

    t0 = time.perf_counter()
    _ = auxf.predict(model, params, DATABASE, scaler_path=scaler_path)
    return time.perf_counter() - t0, params.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tipo", choices=["frecuencia_unica", "espectro_completo"], default="frecuencia_unica")
    parser.add_argument("--modo", choices=["loop", "batch", "ambos"], default="ambos")
    parser.add_argument("--n", type=int, default=None, help="numero de estructuras (por defecto N=100000, o 50 en espectro_completo)")
    args = parser.parse_args()

    n_estructuras = args.n if args.n is not None else (50 if args.tipo == "espectro_completo" else N)

    print(f"Cargando modelo surrogate desde {MODEL_DIR} ...")
    model = models.load_model(MODEL_DIR / "Model_1seed.h5", compile=False)
    scaler_path = MODEL_DIR / "scalers.json"

    # warm-up (traza de tf.function, no cuenta en el tiempo medido)
    _ = auxf.predict(model, np.array([[0.0, D_MIN, D_MIN, FREQ]]), DATABASE, scaler_path=scaler_path)

    print(f"Generando {n_estructuras} estructuras aleatorias (d1, d2 en [{D_MIN},{D_MAX}] nm, theta en [0,180] deg) ...")
    d1_arr, d2_arr, theta_arr = generar_estructuras(n_estructuras)

    if args.tipo == "frecuencia_unica":
        print(f"\n--- TMM: {n_estructuras} simulaciones a {FREQ:.1f} cm-1 ---")
        t_tmm = medir_tmm(d1_arr, d2_arr, theta_arr, FREQ)
        print(f"Tiempo total TMM       : {t_tmm:.3f} s  ({t_tmm / n_estructuras * 1e3:.4f} ms/simulacion)")

        t_nn_loop = None
        if args.modo in ("loop", "ambos"):
            print(f"\n--- Surrogate (1 modelo, llamada a llamada): {n_estructuras} simulaciones a {FREQ:.1f} cm-1 ---")
            t_nn_loop = medir_surrogate_loop(d1_arr, d2_arr, theta_arr, FREQ, model, scaler_path)
            print(f"Tiempo total surrogate (loop)  : {t_nn_loop:.3f} s  ({t_nn_loop / n_estructuras * 1e3:.4f} ms/simulacion)")

        t_nn_batch = None
        if args.modo in ("batch", "ambos"):
            print(f"\n--- Surrogate (1 modelo, batch unico vectorizado): {n_estructuras} simulaciones a {FREQ:.1f} cm-1 ---")
            t_nn_batch = medir_surrogate_batch(d1_arr, d2_arr, theta_arr, FREQ, model, scaler_path)
            print(f"Tiempo total surrogate (batch) : {t_nn_batch:.3f} s  ({t_nn_batch / n_estructuras * 1e3:.4f} ms/simulacion)")

        print()
        if t_nn_loop is not None:
            print(f"Speedup TMM / surrogate(loop)  : {t_tmm / t_nn_loop:.2f}x")
        if t_nn_batch is not None:
            print(f"Speedup TMM / surrogate(batch) : {t_tmm / t_nn_batch:.2f}x")

    else:  # espectro_completo
        freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
        n_evals = n_estructuras * N_FREQS
        print(f"\n--- TMM: {n_estructuras} espectros completos ({N_FREQS} freqs en [{FREQ_MIN},{FREQ_MAX}] cm-1) = {n_evals} evaluaciones ---")
        t_tmm = medir_tmm_espectro(d1_arr, d2_arr, theta_arr, freqs)
        print(f"Tiempo total TMM       : {t_tmm:.3f} s  ({t_tmm / n_estructuras:.4f} s/espectro, {t_tmm / n_evals * 1e3:.5f} ms/eval)")

        print(f"\n--- Surrogate (1 modelo, batch unico vectorizado): {n_estructuras} espectros completos ---")
        t_nn_batch, n_filas = medir_surrogate_batch_espectro(d1_arr, d2_arr, theta_arr, freqs, model, scaler_path)
        print(f"Tiempo total surrogate : {t_nn_batch:.3f} s  ({t_nn_batch / n_estructuras:.4f} s/espectro, {t_nn_batch / n_filas * 1e3:.5f} ms/eval, {n_filas} filas)")

        print(f"\nSpeedup TMM / surrogate (batch) : {t_tmm / t_nn_batch:.2f}x")
