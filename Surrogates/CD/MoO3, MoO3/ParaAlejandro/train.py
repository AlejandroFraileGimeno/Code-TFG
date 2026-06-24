"""
Punto de entrada para ENTRENAR el modelo forward.

Lee el dataset de NN_Code/Dataset_MoO3_Bilayer y guarda el modelo entrenado
en NN_Code/Forward_Models_Trained_bilayers_MoO3/Model_1seed/.
"""

from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))  # permite importar los modulos del proyecto

from training_forward import train_models

directory = str(BASE / "NN_Code" / "Forward_Models_Trained_bilayers_MoO3")
database = str(BASE / "NN_Code" / "Dataset_MoO3_Bilayer")
ntrain = 8000
nvalidation = 2000


if __name__ == "__main__":
    train_models(directory, database, ntrain, nvalidation)
