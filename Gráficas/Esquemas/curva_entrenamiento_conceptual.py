# -*- coding: utf-8 -*-
"""
Curva de entrenamiento conceptual: pérdida frente a tiempo de
entrenamiento para los conjuntos de entrenamiento y validación,
con las zonas de underfitting / ajuste adecuado / overfitting.

Mismos roles de color que las curvas de entrenamiento reales del TFG
(validación en azul, entrenamiento en gris discontinuo).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "cm",
    "font.size":         15,
    "axes.labelsize":    17,
    "axes.titlesize":    18,
    "legend.fontsize":   14,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#c3c2b7",
})

COLOR_VAL   = "#2a78d6"
COLOR_TRAIN = "#898781"

# ----------------------------------------------------------------------
# Curvas conceptuales
# ----------------------------------------------------------------------
t = np.linspace(0, 1, 500)

train = 0.08 + 0.90 * np.exp(-4.5 * t)
val   = 0.14 + 0.90 * np.exp(-4.2 * t) + 0.24 * t ** 2.2

t_opt = t[np.argmin(val)]

# ----------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.9))
ax.set_title("Curva de entrenamiento", fontweight="bold", pad=8)

ax.plot(t, val,   color=COLOR_VAL,   lw=2.2, label="Validación")
ax.plot(t, train, color=COLOR_TRAIN, lw=1.8, ls="--", label="Entrenamiento")

# Línea vertical en el punto de ajuste adecuado (mínimo de validación)
ax.axvline(t_opt, color="#0b0b0b", lw=1.0, ls=":")

# Etiquetas de las zonas
ax.text(0.17, 0.66, "Underfitting", fontsize=15)
ax.text(0.76, 0.42, "Overfitting", fontsize=15)
ax.annotate("Ajuste\nadecuado",
            xy=(t_opt, np.min(val)), xytext=(t_opt - 0.04, 0.50),
            fontsize=15, ha="right", va="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1,
                            shrinkB=4))

# ----------------------------------------------------------------------
# Ejes conceptuales (sin ticks, con flechas)
# ----------------------------------------------------------------------
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xticks([])
ax.set_yticks([])
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
for lado in ("bottom", "left"):
    ax.spines[lado].set_visible(False)

ax.annotate("", xy=(1.05, 0), xytext=(-0.02, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
ax.annotate("", xy=(-0.02, 1.05), xytext=(-0.02, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))

ax.set_xlabel("Tiempo de entrenamiento")
ax.set_ylabel("Valor de la pérdida")

ax.legend(loc="upper right")

fig.tight_layout()

# ----------------------------------------------------------------------
# Guardar
# ----------------------------------------------------------------------
out = Path(__file__).resolve().parents[2] / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fig.savefig(out / "curva_entrenamiento_conceptual.pdf", bbox_inches="tight")
fig.savefig(out / "curva_entrenamiento_conceptual.png", dpi=300, bbox_inches="tight")
print(f"Guardado en: {out}")
plt.show()
