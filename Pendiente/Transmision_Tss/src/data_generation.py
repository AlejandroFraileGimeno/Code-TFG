# -*- coding: utf-8 -*-
"""
===========================================================
data_generation  (proyecto T_ss)
===========================================================
Genera el dataset para aprender la TRANSMITANCIA T_ss = |t_ss|^2 (componente
s->s, polarizacion lineal) de una bicapa de MoO3 con AMBAS capas rotadas, sobre
substrato de BaF2, a incidencia normal (alpha=0).

Formato de salida (lo que lee utils_nn_forward.load_database):
- angles.csv           -> matriz (n_estructuras, 2)  con [theta1, theta2] (grados)
- thickness.csv        -> matriz (n_estructuras, 2)  con [d1, d2] (nm)
- Tss_spectra_norm.csv -> matriz (n_estructuras, n_freqs) con T_ss = |t_ss|^2
                          = calculate_transmission(...)[1]

Estructura (bicapa): [ MoO3(d1, phi=theta1), MoO3(d2, phi=theta2) ]
con superestrato/substrato pasados como argumento (Aire / BaF2 por defecto).
"""

from pathlib import Path
import numpy as np

from generalized_transfer_matrix_method import (
    MoO3,
    LayeredStructure,
    calculate_transmission,
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
    """Genera n_data espectros de transmitancia T_ss para bicapas de MoO3.

    Devuelve
    -------
    Tss_spectra_list : np.ndarray  (n_data, n_freqs)
    angles           : np.ndarray  (n_data, 2)   -> [theta1, theta2] en grados
    thickness        : np.ndarray  (n_data, 2)   -> [d1, d2] en nm
    """
    if not bilayers:
        raise NotImplementedError(
            "Esta version solo cubre el caso bicapa (bilayers=True)."
        )
    if freqs is None:
        raise ValueError("Debes pasar 'freqs' (malla de frecuencias en cm-1).")

    freqs = np.asarray(freqs, dtype=float)
    n_freqs = len(freqs)

    Tss_spectra_list = np.empty((n_data, n_freqs), dtype=float)
    angles = np.empty((n_data, 2), dtype=float)
    thickness = np.empty((n_data, 2), dtype=float)

    i = 0
    n_rechazadas = 0
    while i < n_data:
        d1, d2 = np.random.randint(d_min, d_max + 1, size=2)
        theta1 = np.random.randint(0, 180 + 1)
        theta2 = np.random.randint(0, 180 + 1)

        structure = LayeredStructure(
            superstrate=superstrate_mat,
            substrate=substrate_mat,
            layers=[
                MoO3(d=d1 * 1e-9, phi=np.deg2rad(theta1)),
                MoO3(d=d2 * 1e-9, phi=np.deg2rad(theta2)),
            ],
        )

        espectro = np.empty(n_freqs, dtype=float)
        for j in range(n_freqs):
            t = calculate_transmission(freqs[j], alpha, structure, basis="linear")
            espectro[j] = float(t[1])  # componente s->s

        # Validacion fisica: T_ss en [0,1] y finita. Si no, se RECHAZA y se
        # genera otra estructura (el TMM es inestable en capas gruesas/absorbentes).
        if (not np.all(np.isfinite(espectro))
                or espectro.max() > 1.0 or espectro.min() < 0.0):
            n_rechazadas += 1
            continue

        Tss_spectra_list[i] = espectro
        angles[i, 0] = theta1
        angles[i, 1] = theta2
        thickness[i, 0] = d1
        thickness[i, 1] = d2
        i += 1

        if i % 100 == 0 or i == n_data:
            print(f"  generadas {i}/{n_data} (rechazadas {n_rechazadas})", flush=True)

    return Tss_spectra_list, angles, thickness


def save_dataset(Tss_spectra_list, angles, thickness, output_dir):
    """Guarda el dataset en CSV sin cabecera (formato que lee load_database)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir / "Tss_spectra_norm.csv",
               np.asarray(Tss_spectra_list, dtype=float), delimiter=",")
    np.savetxt(output_dir / "angles.csv",
               np.asarray(angles, dtype=float), delimiter=",")
    np.savetxt(output_dir / "thickness.csv",
               np.asarray(thickness, dtype=float), delimiter=",")

    print(f"Dataset guardado en: {output_dir.resolve()}")
