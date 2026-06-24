"""
Punto de entrada para GENERAR el dataset (version serie, original de Lucia).

OJO: en serie tarda ~3 horas para 10000 estructuras. Para lo mismo pero en
~30 min usa  `python generate_dataset_parallel.py`  (reparte entre nucleos).
Ambos producen los mismos 3 CSV en NN_Code/Dataset_MoO3_Bilayer.
"""

from pathlib import Path
import sys
import numpy as np

from generalized_transfer_matrix_method.permittivities import Air, Au

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from data_generation import generate_data, save_dataset

N = 1000
omega = [600, 1100]
omegas = np.linspace(omega[0], omega[1], N)


if __name__ == "__main__":
    CD_spectra__norm_list, angles, thickness = generate_data(
        n_data=10000,
        bilayers=True,
        superstrate_mat=Air(),
        substrate_mat=Au(),
        freqs=omegas,  # Malla de frecuencias (en cm-1)
        d_min=200,
        d_max=2000,
        alpha=0,  # Angulo de incidencia en grados
        plot_ifc_enabled=False,
        seed_file=None,
    )
    output_dir = BASE / "NN_Code" / "Dataset_MoO3_Bilayer"
    save_dataset(CD_spectra__norm_list, angles, thickness, output_dir)
