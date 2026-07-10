# -*- coding: utf-8 -*-
"""
Esquema conceptual de los enfoques de diseño fotónico, adaptado de
Fig. 1 de Melati et al., APL Photonics 10, 101101 (2025), con la
notación del TFG: parámetros x = (phi_1, d_1, d_2), figura de mérito
f_obj, y el dispositivo representado como la bicapa sobre Au.

(a) Diseño directo
(b) Diseño inverso basado en optimización
(c) Diseño inverso basado en modelo
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        13,
})

COLOR_DESEADA   = "#0b0b0b"   # respuesta objetivo (rol: verdad/objetivo)
COLOR_RESPUESTA = "#e34948"   # respuesta calculada (rol: acento)
AZUL_CLARO      = "#b9d4ef"
GRIS_BORDE      = "0.45"

fig, ax = plt.subplots(figsize=(10.6, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(8, 100)
ax.axis("off")


# ----------------------------------------------------------------------
# Elementos auxiliares
# ----------------------------------------------------------------------
def espectro(x0, y0, w, h, color, doble=False):
    """Mini-espectro con ejes en flecha dentro del rectángulo dado."""
    ax.annotate("", xy=(x0 + w, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
    ax.annotate("", xy=(x0, y0 + h), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
    t = np.linspace(0, 1, 300)
    if doble:
        c = (0.85 * np.exp(-((t - 0.32) / 0.10) ** 2)
             + 0.65 * np.exp(-((t - 0.68) / 0.09) ** 2))
    else:
        c = (0.9 * np.exp(-((t - 0.38) / 0.13) ** 2)
             + 0.25 * np.exp(-((t - 0.75) / 0.07) ** 2))
    ax.plot(x0 + 0.06 * w + t * 0.86 * w,
            y0 + 0.08 * h + c / c.max() * 0.72 * h,
            color=color, lw=1.8, solid_capstyle="round")
    ax.text(x0 + w - 0.5, y0 - 0.8, r"$\omega$", fontsize=10,
            ha="right", va="top")


def dispositivo(x0, y0, w, h):
    """Bicapa sobre sustrato de Au (esquema del TFG)."""
    h3 = h / 3.0
    ax.add_patch(Rectangle((x0, y0), w, h3, facecolor="0.80",
                           edgecolor="black", lw=0.8))
    ax.add_patch(Rectangle((x0, y0 + h3), w, h3, facecolor=AZUL_CLARO,
                           edgecolor="black", lw=0.8))
    ax.add_patch(Rectangle((x0, y0 + 2 * h3), w, h3, facecolor="#7fb2e5",
                           edgecolor="black", lw=0.8, hatch="//"))
    ax.text(x0 + w / 2, y0 + 0.5 * h3, "Au", fontsize=10,
            ha="center", va="center")
    ax.text(x0 + w / 2, y0 + 1.5 * h3, r"$d_2$", fontsize=10,
            ha="center", va="center")
    ax.text(x0 + w / 2, y0 + 2.5 * h3, r"$d_1,\ \phi_1$", fontsize=10,
            ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))


def flecha(x1, y1, x2, y2, estilo="arc3,rad=0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3,
                                connectionstyle=estilo, shrinkA=2, shrinkB=2))


def panel(x0, y0, w, h):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none",
                           edgecolor=GRIS_BORDE, lw=1.0))


def cabecera(xc, yc, titulo_txt, subtitulo=None):
    ax.text(xc, yc, titulo_txt, fontsize=11.5, fontweight="bold",
            ha="center", va="center")
    if subtitulo:
        ax.text(xc, yc - 4.6, subtitulo, fontsize=9.5,
                ha="center", va="center")


# ----------------------------------------------------------------------
# (a) Diseño directo  — arriba a la izquierda
# ----------------------------------------------------------------------
panel(1, 60, 45, 39)
ax.text(2.5, 96.4, "(a) Diseño directo", fontsize=13, fontweight="bold",
        ha="left", va="center")

cabecera(10.5, 89.5, "Descripción del\ndispositivo", "(geometría, material...)")
dispositivo(4.5, 71, 12, 12)
ax.text(10.5, 67.8, "Parámetros", fontsize=10.5, ha="center", va="top")
ax.text(10.5, 64.5, r"$\mathbf{x} = (\phi_1,\ d_1,\ d_2)$", fontsize=11.5,
        ha="center", va="top")

flecha(18.5, 77.5, 28.5, 77.5)
ax.text(23.5, 79.5, "Simulación,", fontsize=10.5, ha="center", va="bottom")
ax.text(23.5, 75.7, "surrogate...", fontsize=10.5, ha="center", va="top",
        style="italic")

cabecera(36.5, 89.5, "Respuesta del\ndispositivo", "(espectro...)")
espectro(30.5, 71, 13, 12, COLOR_RESPUESTA)
ax.text(36.5, 67.8, "Figura de mérito", fontsize=10.5, ha="center", va="top")
ax.text(36.5, 64.5, r"$f_{\mathrm{obj}}(\mathbf{x})$", fontsize=11.5,
        ha="center", va="top")

# ----------------------------------------------------------------------
# (c) Diseño inverso basado en modelo — abajo a la izquierda
# ----------------------------------------------------------------------
panel(1, 9, 45, 47)
ax.text(2.5, 53.2, "(c) Diseño inverso basado en modelo", fontsize=11.5,
        fontweight="bold", ha="left", va="center")

cabecera(10.5, 46.5, "Respuesta\ndeseada", "(espectro...)")
espectro(4.5, 28, 13, 12, COLOR_DESEADA, doble=True)
ax.text(10.5, 24.8, "Figura de mérito", fontsize=10.5, ha="center", va="top")
ax.text(10.5, 21.5, r"deseada, $\bar{f}_{\mathrm{obj}}$", fontsize=11.5,
        ha="center", va="top")

flecha(18.5, 34.5, 28.6, 34.5)
ax.text(23.0, 36.5, "Modelo inverso", fontsize=10, ha="center", va="bottom")
ax.text(23.0, 32.7, "entrenado", fontsize=10, ha="center", va="top")

cabecera(36, 46.5, "Descripción del\ndispositivo", "(geometría, material...)")
dispositivo(30, 28, 12, 12)
ax.text(36, 24.8, "Parámetros óptimos", fontsize=10.5, ha="center", va="top")
ax.text(36, 21.5, r"$\bar{\mathbf{x}} = (\bar{\phi}_1,\ \bar{d}_1,\ \bar{d}_2)$",
        fontsize=11.5, ha="center", va="top")

# ----------------------------------------------------------------------
# (b) Diseño inverso basado en optimización — derecha (grande)
# ----------------------------------------------------------------------
panel(49, 9, 50, 90)
ax.text(74, 94.7, "(b) Diseño inverso basado\nen optimización",
        fontsize=13, fontweight="bold", ha="center", va="center")

# Respuesta deseada (arriba izquierda del panel)
cabecera(58.5, 83.5, "Respuesta\ndeseada", "(espectro...)")
espectro(52.5, 66.5, 12, 11, COLOR_DESEADA, doble=True)
ax.text(58.5, 63.3, "Figura de mérito", fontsize=10.5, ha="center", va="top")
ax.text(58.5, 60.0, r"deseada, $\bar{f}_{\mathrm{obj}}$", fontsize=11.5,
        ha="center", va="top")

# Caja del algoritmo de optimización (centro)
ax.add_patch(FancyBboxPatch((68.5, 68), 13.5, 7.5,
                            boxstyle="round,pad=0.6",
                            facecolor="white", edgecolor="black", lw=1.2))
ax.text(75.25, 71.75, "Algoritmo de\noptimización", fontsize=11.5,
        ha="center", va="center")

flecha(65.3, 71.75, 67.6, 71.75)

# Dispositivo (arriba derecha)
cabecera(90.5, 83.5, "Descripción del\ndispositivo")
dispositivo(84.5, 66.5, 12, 12)
ax.text(90.5, 63.3, "Parámetros", fontsize=10.5, ha="center", va="top")
ax.text(90.5, 60.0, r"$\mathbf{x} = (\phi_1,\ d_1,\ d_2)$", fontsize=11.5,
        ha="center", va="top")

flecha(82.9, 71.75, 84.2, 71.75)

# Simulación (bajada por la derecha)
flecha(90.5, 56, 90.5, 44.5)
ax.text(89.2, 51.7, "Simulación,", fontsize=10.5, ha="right", va="center")
ax.text(89.2, 48.7, "surrogate...", fontsize=10.5, ha="right", va="center",
        style="italic")

# Respuesta del dispositivo (abajo derecha, bajo el dispositivo)
cabecera(90.5, 40, "Respuesta del\ndispositivo", "(espectro...)")
espectro(84.5, 20.5, 12, 12, COLOR_RESPUESTA)
ax.text(90.5, 17.3, "Figura de mérito", fontsize=10.5, ha="center", va="top")
ax.text(90.5, 14.0, r"$f_{\mathrm{obj}}(\mathbf{x})$", fontsize=11.5,
        ha="center", va="top")

# Cierre del bucle: respuesta -> algoritmo de optimización
flecha(83.6, 26.5, 75.25, 67.2, estilo="angle,angleA=180,angleB=90,rad=5")
ax.text(73.6, 47, "Actualización\nde parámetros", fontsize=10.5,
        ha="right", va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

# ----------------------------------------------------------------------
# Guardar
# ----------------------------------------------------------------------
out = Path(__file__).resolve().parents[2] / "Arreglos en Gráficos"
out.mkdir(exist_ok=True)
fig.savefig(out / "esquema_diseno_inverso.pdf", bbox_inches="tight")
fig.savefig(out / "esquema_diseno_inverso.png", dpi=300, bbox_inches="tight")
print(f"Guardado en: {out}")
plt.show()
