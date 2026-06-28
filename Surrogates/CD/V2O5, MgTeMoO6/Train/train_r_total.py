import sys
import time
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))

from training_r_total import train_r_total

database  = str(ROOT_PATH / "Datasets" / "CD" / "V2O5_MgTeMoO6")
directory = str(ROOT_PATH / "Models"   / "R_total" / "V2O5_MgTeMoO6")

seeds = [727402, 37847, 56789, 98765, 11223]

for i, seed in enumerate(seeds, 1):
    print(f"\n===== Modelo {i}/5  seed={seed} =====")
    t0 = time.time()
    train_r_total(
        database=database,
        output_dir=directory + f"/Model_{i}seed",
        ntrain=8000, nvalidation=2000, seed=seed,
    )
    print(f"---Training completed in {time.time()-t0:.5f} seconds ---")

print("\nAll 5 R_total models trained.")