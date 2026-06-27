from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_forward import train_models

directory   = ROOT_PATH / "Models" / "CD" / "MoO3_MgTeMoO6"
database    = ROOT_PATH / "Datasets" / "CD" / "MoO3_MgTeMoO6"
ntrain      = 8000
nvalidation = 2000

train_models(str(directory), str(database), ntrain, nvalidation)