# -*- coding: utf-8 -*-
"""
Historial de entrenamiento — Surrogates CD y R_total (MoO3/MoO3)
Genera una figura PNG por magnitud (|CD| y R_total) con las curvas del
primer modelo (Model_1seed), en el mismo estilo que los filtros ópticos
(Tandem/T_xx/plot_training_history.py).
Formato de los .txt: epoch  val_loss  train_loss
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Estilo publicación TFG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":          "serif",
    "mathtext.fontset":     "cm",
    "font.size":            15,
    "axes.labelsize":       16,
    "axes.titlesize":       16,
    "xtick.labelsize":      14,
    "ytick.labelsize":      14,
    "axes.linewidth":       0.9,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "legend.fontsize":      13,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#c3c2b7",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.35,
    "grid.linestyle":       "--",
})

COLOR_VAL   = "#2a78d6"
COLOR_TRAIN = "#898781"

ROOT       = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "Models"
BASE_DIR   = Path(__file__).resolve().parent

CASOS = [
    (r"$|\mathbf{CD}_{\mathbf{norm}}|$",
     MODELS_DIR / "CD" / "MoO3_MoO3",
     BASE_DIR / "Evaluación" / "training_history_CD.png"),
    (r"$\mathbf{R}_{\mathbf{total}}$",
     MODELS_DIR / "R_total" / "MoO3_MoO3",
     BASE_DIR / "Evaluación R_total" / "training_history_R_total.png"),
]


for titulo, model_dir, out in CASOS:
    hist_file = model_dir / "Model_1seed" / "history_loss.txt"
    if not hist_file.exists():
        print(f"Sin datos: {hist_file}")
        continue
    hist = np.loadtxt(hist_file)

    epochs     = hist[:, 0]
    val_mean   = hist[:, 1]
    train_mean = hist[:, 2]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.semilogy(epochs, val_mean,   color=COLOR_VAL,   lw=2.0, label="Validación")
    ax.semilogy(epochs, train_mean, color=COLOR_TRAIN, lw=1.4, ls="--", label="Entrenamiento")

    mse_final = val_mean[-1]
    mantissa, exp = f"{mse_final:.2e}".split("e")
    exp_int = int(exp)
    mse_latex = rf"{mantissa} \cdot 10^{{{exp_int}}}"
    ax.axhline(mse_final, color=COLOR_VAL, lw=0.7, ls=":", alpha=0.6)
    ax.text(0.97, 0.12,
            f"$\\mathrm{{MSE}}_{{\\mathrm{{val}}}} = {mse_latex}$",
            ha="right", va="bottom", fontsize=16,
            color=COLOR_VAL,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE (escala logarítmica)")
    ax.set_title(titulo, pad=9, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, which="both")
    ax.set_xlim(left=0)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="auto", numticks=10))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Guardado: {out}")
    plt.close(fig)
