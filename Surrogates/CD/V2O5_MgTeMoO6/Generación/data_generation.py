import sys
from pathlib import Path
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, Au, V2O5, MgTeMoO6, LayeredStructure, calculate_circular_dichroism_ref,
)


def generate_data(d1, d2, theta, omega, alpha=0):
    structure = LayeredStructure(
        superstrate=Air(), substrate=Au(),
        layers=[
            V2O5(d=d1 * 1e-9, phi=np.deg2rad(theta)),
            MgTeMoO6(d=d2 * 1e-9),
        ],
    )
    results = [calculate_circular_dichroism_ref(f, alpha, structure) for f in omega]
    cd_norm  = [abs(r[1]) for r in results]
    r_total  = [r[2]      for r in results]
    return cd_norm, r_total