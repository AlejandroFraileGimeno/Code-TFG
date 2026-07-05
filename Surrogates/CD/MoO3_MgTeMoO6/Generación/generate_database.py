from pathlib import Path
import sys
import numpy as np

TMM_PATH = Path(__file__).resolve().parents[4] / "TMM"
sys.path.insert(0, str(TMM_PATH))

from generalized_transfer_matrix_method.permittivities import Air, Au

from data_generation import generate_data, save_dataset


N = 1000
omega = [600, 1100]
omegas = np.linspace(omega[0], omega[1], N)

CD_spectra_norm_list, R_total_spectra_list, angles, thickness = generate_data(
    n_data=10000,
    bilayers=True,
    superstrate_mat=Air(),
    substrate_mat=Au(),
    freqs=omegas,
    d_min=200,
    d_max=2000,
    alpha=0,
    plot_ifc_enabled=False,
    seed_file=Path(__file__).resolve().parents[4] / "Seed" / "SEED_LIST.csv",
)
output_dir = Path(__file__).resolve().parents[4] / "Datasets" / "CD" / "MoO3_MgTeMoO6"
save_dataset(CD_spectra_norm_list, R_total_spectra_list, angles, thickness, output_dir)