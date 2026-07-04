# -*- coding: utf-8 -*-
"""
Historial de entrenamiento — Forward models T_xx
Genera una figura PNG por cada par de materiales.
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
ALPHA_BAND  = 0.18

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "Models" / "T_xx"
OUT_DIR    = Path(__file__).resolve().parent

TANDEM_DIR = Path(__file__).resolve().parent

PARES = [
    ("MoO$_3$ / V$_2$O$_5$",
     MODELS_DIR / "MoO3_V2O5_BaF2" / "Forward",
     TANDEM_DIR / "MoO3_V2O5_BaF2" / "Evaluación" / "training_history.png"),
    ("V$_2$O$_5$ / MoO$_3$",
     MODELS_DIR / "V2O5_MoO3_BaF2" / "Forward",
     TANDEM_DIR / "V2O5_MoO3_BaF2" / "Evaluación" / "training_history.png"),
]


def load_histories(model_dir: Path):
    files = sorted(model_dir.glob("Model_*seed/history_loss.txt"))
    return [np.loadtxt(f) for f in files]


for titulo, model_dir, out in PARES:
    histories = load_histories(model_dir)
    if not histories:
        print(f"Sin datos: {model_dir}")
        continue

    min_ep    = min(h.shape[0] for h in histories)
    epochs    = histories[0][:min_ep, 0]
    val_mat   = np.stack([h[:min_ep, 1] for h in histories])
    train_mat = np.stack([h[:min_ep, 2] for h in histories])

    val_mean   = val_mat.mean(0);  val_lo  = val_mat.min(0);  val_hi  = val_mat.max(0)
    train_mean = train_mat.mean(0); train_lo = train_mat.min(0); train_hi = train_mat.max(0)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.fill_between(epochs, val_lo,   val_hi,   color=COLOR_VAL,   alpha=ALPHA_BAND)
    ax.fill_between(epochs, train_lo, train_hi, color=COLOR_TRAIN, alpha=ALPHA_BAND)

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
            transform=ax.transAxes)

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE (escala logarítmica)")
    ax.set_title(titulo, pad=9)
    ax.legend(loc="upper right")
    ax.grid(True, which="both")
    ax.set_xlim(left=0)
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="auto", numticks=10))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Guardado: {out}")
    plt.show()
