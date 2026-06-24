"""
Punto de entrada para GENERAR el dataset T_ss (version serie).

OJO: en serie tarda mucho (~3 h para 10000 estructuras). Para lo mismo pero en
~30 min usa  `python generate_dataset_parallel.py`  (reparte entre nucleos).
Ambos producen los mismos 3 CSV en NN_Code/Dataset_Tss_Bilayer.
"""

from pathlib import Path
import sys
import numpy as np

from generalized_transfer_matrix_method.permittivities import Air, BaF2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from data_generation import generate_data, save_dataset

N = 1000
omega = [600, 1100]
omegas = np.linspace(omega[0], omega[1], N)
ALPHA = np.deg2rad(40)  # angulo de incidencia (40°, como el notebook)


if __name__ == "__main__":
    Tss_spectra_list, angles, thickness = generate_data(
        n_data=10000,
        bilayers=True,
        superstrate_mat=Air(),
        substrate_mat=BaF2(),   # substrato de BaF2 (transmision)
        freqs=omegas,  # Malla de frecuencias (en cm-1)
        d_min=200,
        d_max=2000,
        alpha=ALPHA,  # Angulo de incidencia (radianes)
        plot_ifc_enabled=False,
        seed_file=None,
    )
    output_dir = BASE / "NN_Code" / "Dataset_Tss_Bilayer"
    save_dataset(Tss_spectra_list, angles, thickness, output_dir)
