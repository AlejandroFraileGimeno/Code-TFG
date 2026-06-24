"""
Grafica la curva de entrenamiento (pérdida) del modelo entrenado.

Lee history_loss.txt (columnas: epoca  val_loss  train_loss) y guarda un PNG
junto al modelo. Pulsa F5 con este archivo abierto para verlo.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "NN_Code" / "Forward_Models_Trained_bilayers_MoO3" / "Model_1seed"
hist_file = MODEL_DIR / "history_loss.txt"

data = np.loadtxt(hist_file)
epochs, val_loss, train_loss = data[:, 0], data[:, 1], data[:, 2]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Entrenamiento (train)")
plt.plot(epochs, val_loss, label="Validación (val)")
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("Pérdida (Huber, escala log)")
plt.title("Curva de entrenamiento del modelo forward")
plt.legend()
plt.grid(True, which="both", alpha=0.3)

out = MODEL_DIR / "curva_entrenamiento.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Mejor val_loss: {val_loss.min():.5f} (época {int(epochs[val_loss.argmin()])})")
print(f"Gráfica guardada en: {out}")
plt.show()
