# -*- coding: utf-8 -*-
"""
===========================================================
data_generation  (RECONSTRUIDO)
===========================================================
NOTA: el modulo original de Lucia (`src/data_generation.py`) no estaba en el
equipo. Este fichero se ha reconstruido a partir del resto del codigo
(`forward_model.run_inference`, `utils_nn_forward.load_database/expand_dataset`
y `optics.calculate_circular_dichroism_ref`) para reproducir EXACTAMENTE el
formato de dataset que espera el pipeline:

- angles.csv          -> matriz (n_estructuras, 1)  con el angulo theta (grados)
- thickness.csv       -> matriz (n_estructuras, 2)  con [d1, d2] (nm)
- CD_spectra_norm.csv -> matriz (n_estructuras, n_freqs) con |CD_norm|
                         donde CD_norm = (R_r - R_l) / (R_r + R_l)
                         es decir abs(calculate_circular_dichroism_ref(...)[1])
                         (el mismo valor que usa run_inference como "CD true").
- R_total_spectra.csv -> matriz (n_estructuras, n_freqs) con R_total = R_r + R_l
                         es decir calculate_circular_dichroism_ref(...)[2].

Estructura (bicapa): [ MoO3(d1, phi=theta), MoO3(d2, phi=0) ]
con superestrato/substrato pasados como argumento (Air / Au por defecto).
"""

from pathlib import Path
import numpy as np

from generalized_transfer_matrix_method import (
    MoO3,
    LayeredStructure,
    calculate_circular_dichroism_ref,
)


def generate_data(
    n_data,
    bilayers=True,
    superstrate_mat=None,
    substrate_mat=None,
    freqs=None,
    d_min=200,
    d_max=2000,
    alpha=0,
    plot_ifc_enabled=False,
    seed_file=None,
):
    """Genera n_data espectros de dicroismo circular (CD) para bicapas de MoO3.

    Devuelve
    -------
    CD_spectra_norm_list : np.ndarray  (n_data, n_freqs)
    angles               : np.ndarray  (n_data, 1)   -> theta en grados
    thickness            : np.ndarray  (n_data, 2)   -> [d1, d2] en nm
    """
    if not bilayers:
        raise NotImplementedError(
            "Esta reconstruccion solo cubre el caso bicapa (bilayers=True)."
        )
    if freqs is None:
        raise ValueError("Debes pasar 'freqs' (malla de frecuencias en cm-1).")

    freqs = np.asarray(freqs, dtype=float)
    n_freqs = len(freqs)

    seeds = np.loadtxt(seed_file, dtype=int) if seed_file is not None else None

    CD_spectra_norm_list = np.empty((n_data, n_freqs), dtype=float)
    R_total_spectra_list = np.empty((n_data, n_freqs), dtype=float)
    angles = np.empty((n_data, 1), dtype=float)
    thickness = np.empty((n_data, 2), dtype=float)

    for i in range(n_data):
        if seeds is not None:
            np.random.seed(seeds[i])
        d1, d2 = np.random.randint(d_min, d_max + 1, size=2)
        theta = np.random.randint(0, 180 + 1)

        structure = LayeredStructure(
            superstrate=superstrate_mat,
            substrate=substrate_mat,
            layers=[
                MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta)),
                MoO3(d=d2 * 1e-9),
            ],
        )

        for j in range(n_freqs):
            cd = calculate_circular_dichroism_ref(freqs[j], alpha, structure)
            CD_spectra_norm_list[i, j] = abs(cd[1])
            R_total_spectra_list[i, j] = cd[2]

        angles[i, 0] = theta
        thickness[i, 0] = d1
        thickness[i, 1] = d2

        if (i + 1) % 100 == 0 or (i + 1) == n_data:
            print(f"  generadas {i + 1}/{n_data} estructuras", flush=True)

    return CD_spectra_norm_list, R_total_spectra_list, angles, thickness


def save_dataset(CD_spectra_norm_list, R_total_spectra_list, angles, thickness, output_dir):
    """Guarda el dataset en CSV sin cabecera (formato que lee load_database)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir / "CD_spectra_norm.csv",
               np.asarray(CD_spectra_norm_list, dtype=float), delimiter=",")
    np.savetxt(output_dir / "R_total_spectra.csv",
               np.asarray(R_total_spectra_list, dtype=float), delimiter=",")
    np.savetxt(output_dir / "angles.csv",
               np.asarray(angles, dtype=float), delimiter=",")
    np.savetxt(output_dir / "thickness.csv",
               np.asarray(thickness, dtype=float), delimiter=",")

    print(f"Dataset guardado en: {output_dir.resolve()}")
