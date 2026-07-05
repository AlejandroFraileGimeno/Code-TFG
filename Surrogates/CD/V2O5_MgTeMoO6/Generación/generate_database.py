import sys
import time
from pathlib import Path
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).parent))

from data_generation import generate_data

OUTPUT_DIR = ROOT_PATH / "Datasets" / "CD" / "V2O5_MgTeMoO6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

omega  = np.linspace(400, 1100, 1000)
N      = 1000
n_data = 10000

d1_vals    = np.random.randint(200, 2001, size=n_data)
d2_vals    = np.random.randint(200, 2001, size=n_data)
theta_vals = np.random.randint(0, 181,   size=n_data)

cd_rows = np.empty((n_data, N), dtype=float)
rt_rows = np.empty((n_data, N), dtype=float)
t0 = time.time()

for i in range(n_data):
    cd, rt = generate_data(d1_vals[i], d2_vals[i], theta_vals[i], omega)
    cd_rows[i] = cd
    rt_rows[i] = rt
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"[{i+1:5d}/{n_data}]  {elapsed:.1f}s  ETA {elapsed/(i+1)*(n_data-i-1):.0f}s")

angles    = theta_vals.reshape(-1, 1).astype(float)
thickness = np.column_stack([d1_vals, d2_vals]).astype(float)

np.savetxt(OUTPUT_DIR / "CD_spectra_norm.csv", cd_rows,   delimiter=",")
np.savetxt(OUTPUT_DIR / "R_total_spectra.csv", rt_rows,   delimiter=",")
np.savetxt(OUTPUT_DIR / "angles.csv",          angles,    delimiter=",")
np.savetxt(OUTPUT_DIR / "thickness.csv",       thickness, delimiter=",")

print(f"Dataset guardado en {OUTPUT_DIR}  ({time.time()-t0:.1f}s total)")