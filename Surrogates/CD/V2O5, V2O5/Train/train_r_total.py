from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_r_total import train_models

directory   = ROOT_PATH / "Models" / "R_total" / "V2O5_V2O5"
database    = ROOT_PATH / "Datasets" / "CD" / "V2O5_V2O5"
ntrain      = 8000
nvalidation = 2000

train_models(str(directory), str(database), ntrain, nvalidation)