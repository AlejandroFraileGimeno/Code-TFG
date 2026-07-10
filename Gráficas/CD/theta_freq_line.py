# -*- coding: utf-8 -*-
"""
Extrae la cresta de maximo CD de un water-plot y representa:
  1. Mapa CD(phi, omega) con la cresta superpuesta.
  2. CD evaluado a lo largo de la cresta.
  3. Reflectancia evaluada a lo largo de la cresta.

La salida esta pensada para una memoria de TFG: figura limpia, paneles
etiquetados y exportacion en PNG y PDF.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# CONFIGURACION
# ============================================================================
PAIR = "MoO3_V2O5"
D1_NM = 400
D2_NM = 900

PHI_MIN = 35
PHI_MAX = 85
THETA_STEP = 1

USE_TMM = True
NUM_SEEDS = 1

# Si ejecutas este archivo fuera de la estructura original del proyecto,
# escribe aqui la ruta raiz. Si lo dejas en None se conserva el comportamiento
# del script original: dos carpetas por encima de este archivo.
PROJECT_ROOT: Optional[Path] = None


# ============================================================================
# ESTILO GRAFICO
# ============================================================================
PANEL_LABELS = ("a", "b", "c")
FIGURE_TITLE = "Variación φ MoO₃/V₂O₅"
COLOR_CD = "#1f77b4"
COLOR_RR = "#d62728"
COLOR_RL = "#1f77b4"
COLOR_RT = "#6f6f6f"
COLOR_FIT = "#8ecae6"


def apply_tfg_style() -> None:
    """Ajustes globales de Matplotlib para una figura de memoria."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.fontsize": 14,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#d7d7d7",
            "grid.color": "#c9c9c9",
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
        }
    )


def project_root() -> Path:
    if PROJECT_ROOT is not None:
        return PROJECT_ROOT.expanduser().resolve()

    current_file = Path(__file__).resolve()
    for folder in (current_file.parent, *current_file.parents):
        if (folder / "TMM").is_dir() and (folder / "Models").is_dir():
            return folder

    return Path(__file__).resolve().parents[2]


ROOT_PATH = project_root()
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (  # noqa: E402
    Air,
    Au,
    LayeredStructure,
    MgTeMoO6,
    MoO3,
    V2O5,
    calculate_circular_dichroism_ref,
)


SURROGATE_MAP = {
    "MgTeMoO6_MgTeMoO6": "MgTeMoO6_MgTeMoO6",
    "MoO3_MgTeMoO6": "MoO3_MgTeMoO6",
    "MoO3_MoO3": "MoO3_MoO3",
    "MoO3_V2O5": "MoO3_V2O5",
    "V2O5_V2O5": "V2O5_V2O5",
    "V2O5_MgTeMoO6": "V2O5_MgTeMoO6",
    "MgTeMoO6_MoO3": "MgTeMoO6_MoO3",
    "MgTeMoO6_V2O5": "MgTeMoO6_V2O5",
    "V2O5_MoO3": "V2O5_MoO3",
}


def build_layers(pair: str, d1_m: float, d2_m: float, phi_rad: float):
    layers = {
        "MgTeMoO6_MgTeMoO6": (MgTeMoO6(d=d1_m, phi=phi_rad), MgTeMoO6(d=d2_m)),
        "MoO3_MgTeMoO6": (MoO3(d=d1_m, phi=phi_rad), MgTeMoO6(d=d2_m)),
        "MoO3_MoO3": (MoO3(d=d1_m, phi=phi_rad), MoO3(d=d2_m)),
        "MoO3_V2O5": (MoO3(d=d1_m, phi=phi_rad), V2O5(d=d2_m)),
        "V2O5_V2O5": (V2O5(d=d1_m, phi=phi_rad), V2O5(d=d2_m)),
        "V2O5_MgTeMoO6": (V2O5(d=d1_m, phi=phi_rad), MgTeMoO6(d=d2_m)),
        "MgTeMoO6_MoO3": (MgTeMoO6(d=d1_m, phi=phi_rad), MoO3(d=d2_m)),
        "MgTeMoO6_V2O5": (MgTeMoO6(d=d1_m, phi=phi_rad), V2O5(d=d2_m)),
        "V2O5_MoO3": (V2O5(d=d1_m, phi=phi_rad), MoO3(d=d2_m)),
    }
    return layers[pair]


def load_frequency_grid(root: Path) -> np.ndarray:
    model_dir = root / "Models" / "CD" / PAIR / "Model_1seed"
    with (model_dir / "scalers.json").open("r", encoding="utf-8") as file:
        scaler_ref = json.load(file)

    freq_min = scaler_ref["feature_min"][3]
    freq_max = scaler_ref["feature_max"][3]
    n_freqs = scaler_ref["n_freqs"]
    return np.linspace(freq_min, freq_max, n_freqs)


def load_surrogate_models(root: Path):
    sys.path.insert(0, str(root / "Surrogates" / "CD" / SURROGATE_MAP[PAIR]))

    import utils_nn_forward as auxf  # noqa: PLC0415
    from tensorflow.keras import models as tf_models  # noqa: PLC0415

    cd_model_dir = root / "Models" / "CD" / PAIR
    rt_model_dir = root / "Models" / "R_total" / PAIR

    models_cd, scalers_cd = [], []
    models_rt, scalers_rt = [], []

    for seed in range(1, NUM_SEEDS + 1):
        model_name = f"Model_{seed}seed"

        models_cd.append(
            tf_models.load_model(
                cd_model_dir / model_name / f"{model_name}.h5",
                compile=False,
            )
        )
        scalers_cd.append(str(cd_model_dir / model_name / "scalers.json"))

        models_rt.append(
            tf_models.load_model(
                rt_model_dir / model_name / f"{model_name}.h5",
                compile=False,
            )
        )
        scalers_rt.append(str(rt_model_dir / model_name / "scalers.json"))

    return auxf, models_cd, scalers_cd, models_rt, scalers_rt


def compute_with_tmm(freqs: np.ndarray, thetas: np.ndarray):
    cd_matrix = np.zeros((len(thetas), len(freqs)))
    r_matrix = np.zeros_like(cd_matrix)
    rr_matrix = np.zeros_like(cd_matrix)
    rl_matrix = np.zeros_like(cd_matrix)

    for idx, theta in enumerate(thetas):
        phi_rad = np.deg2rad(theta)
        layer_1, layer_2 = build_layers(PAIR, D1_NM * 1e-9, D2_NM * 1e-9, phi_rad)
        structure = LayeredStructure(
            superstrate=Air(),
            substrate=Au(),
            layers=[layer_1, layer_2],
        )

        results = [calculate_circular_dichroism_ref(freq, 0, structure) for freq in freqs]
        cd_matrix[idx] = np.array([abs(row[1]) for row in results])
        r_matrix[idx] = np.array([row[2] for row in results])
        rr_matrix[idx] = np.array([row[3] for row in results])
        rl_matrix[idx] = np.array([row[4] for row in results])

    return cd_matrix, r_matrix, rr_matrix, rl_matrix


def compute_with_nn(root: Path, freqs: np.ndarray, thetas: np.ndarray):
    auxf, models_cd, scalers_cd, models_rt, scalers_rt = load_surrogate_models(root)

    database_cd = str(root / "Datasets" / "CD" / PAIR)
    database_rt = str(root / "Datasets" / "R_total" / PAIR)

    cd_matrix = np.zeros((len(thetas), len(freqs)))
    r_matrix = np.zeros_like(cd_matrix)

    for idx, theta in enumerate(thetas):
        params_batch = np.column_stack(
            [
                np.full(len(freqs), theta),
                np.full(len(freqs), D1_NM),
                np.full(len(freqs), D2_NM),
                freqs,
            ]
        )

        preds_cd, preds_rt = [], []

        for model, scaler_path in zip(models_cd, scalers_cd):
            batch, _ = auxf.predict(
                model,
                params_batch,
                database_cd,
                scaler_path=scaler_path,
            )
            preds_cd.append([abs(float(np.squeeze(value))) for value in batch])

        for model, scaler_path in zip(models_rt, scalers_rt):
            batch, _ = auxf.predict(
                model,
                params_batch,
                database_rt,
                scaler_path=scaler_path,
            )
            preds_rt.append([abs(float(np.squeeze(value))) for value in batch])

        cd_matrix[idx] = np.mean(preds_cd, axis=0)
        r_matrix[idx] = np.mean(preds_rt, axis=0)

    return cd_matrix, r_matrix, None, None


def extract_ridge(
    freqs: np.ndarray,
    thetas: np.ndarray,
    cd_matrix: np.ndarray,
    r_matrix: np.ndarray,
    rr_matrix: Optional[np.ndarray],
    rl_matrix: Optional[np.ndarray],
):
    mask_phi = (thetas >= PHI_MIN) & (thetas <= PHI_MAX)
    thetas_sel = thetas[mask_phi]
    cd_sel = cd_matrix[mask_phi]
    r_sel = r_matrix[mask_phi]

    ridge_idx = np.nanargmax(cd_sel, axis=1)
    row_idx = np.arange(len(thetas_sel))

    ridge = {
        "theta": thetas_sel,
        "index": ridge_idx,
        "freq": freqs[ridge_idx],
        "cd": cd_sel[row_idx, ridge_idx],
        "r_total": r_sel[row_idx, ridge_idx],
    }

    if rr_matrix is not None and rl_matrix is not None:
        rr_sel = rr_matrix[mask_phi]
        rl_sel = rl_matrix[mask_phi]
        ridge["r_r"] = rr_sel[row_idx, ridge_idx]
        ridge["r_l"] = rl_sel[row_idx, ridge_idx]

    return ridge


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.015,
        0.96,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )


def add_frequency_axis(ax, theta: np.ndarray, freq: np.ndarray) -> None:
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())

    phi_ticks = np.linspace(theta[0], theta[-1], 6)
    freq_ticks = np.interp(phi_ticks, theta, freq)

    top_ax.set_xticks(phi_ticks)
    top_ax.set_xticklabels([f"{value:.0f}" for value in freq_ticks])
    top_ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    top_ax.tick_params(direction="in")


def save_ridge_table(out_dir: Path, ridge: dict[str, np.ndarray], mode: str) -> Path:
    out_path = out_dir / f"ridge_{PAIR}_d1{D1_NM}_d2{D2_NM}_{mode}.csv"

    columns = ["phi_deg", "omega_cm-1", "abs_cd_norm", "r_total"]
    data = [ridge["theta"], ridge["freq"], ridge["cd"], ridge["r_total"]]

    if "r_r" in ridge and "r_l" in ridge:
        columns.extend(["r_r", "r_l"])
        data.extend([ridge["r_r"], ridge["r_l"]])

    table = np.column_stack(data)
    np.savetxt(
        out_path,
        table,
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.8g",
    )
    return out_path


def make_figure(
    freqs: np.ndarray,
    thetas: np.ndarray,
    cd_matrix: np.ndarray,
    ridge: dict[str, np.ndarray],
    mode: str,
):
    apply_tfg_style()

    coeffs = np.polyfit(ridge["theta"], ridge["freq"], 1)
    freq_fit = np.polyval(coeffs, ridge["theta"])

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 9.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0]},
    )
    fig.suptitle(FIGURE_TITLE, fontsize=18, fontweight="bold", y=1.02)
    ax_map, ax_cd, ax_r = axes

    # Panel 1: water-plot y cresta.
    vmax = np.nanmax(cd_matrix)
    im = ax_map.pcolormesh(
        freqs,
        thetas,
        cd_matrix,
        cmap="magma",
        shading="auto",
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )
    cbar = fig.colorbar(im, ax=ax_map, pad=0.015, aspect=28)
    cbar.set_label(r"$|CD_{\mathrm{norm}}|$", fontsize=18)

    ax_map.axhspan(PHI_MIN, PHI_MAX, color="white", alpha=0.06, lw=0)
    ax_map.plot(
        freq_fit,
        ridge["theta"],
        "--",
        color=COLOR_FIT,
        lw=1.5,
        label="Ajuste lineal",
    )


    ax_map.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax_map.set_ylabel(r"$\phi_1$ ($^\circ$)")
    ax_map.set_yticks(np.arange(0, 181, 30))
    add_panel_label(ax_map, PANEL_LABELS[0])

    # Panel 2: CD sobre la cresta.
    ax_cd.plot(
        ridge["theta"],
        ridge["cd"],
        "o-",
        ms=4,
        lw=1.5,
        color=COLOR_CD,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=r"$|CD_{\mathrm{norm}}|$",
    )
    ax_cd.axhline(
        np.mean(ridge["cd"]),
        color=COLOR_RT,
        ls="--",
        lw=1.0,
        label=rf"Media = {np.mean(ridge['cd']):.2f}",
    )
    ax_cd.set_ylabel(r"$|CD_{\mathrm{norm}}|$")
    ax_cd.set_ylim(0, 1.02)
    ax_cd.grid(True)
    ax_cd.legend(loc="best")
    add_panel_label(ax_cd, PANEL_LABELS[1])
    add_frequency_axis(ax_cd, ridge["theta"], ridge["freq"])

    # Panel 3: reflectancia sobre la cresta.
    if "r_r" in ridge and "r_l" in ridge:
        ax_r.plot(
            ridge["theta"],
            ridge["r_r"],
            "o-",
            ms=4,
            lw=1.5,
            color=COLOR_RR,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=r"$R_r$",
        )
        ax_r.plot(
            ridge["theta"],
            ridge["r_l"],
            "o-",
            ms=4,
            lw=1.5,
            color=COLOR_RL,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=r"$R_l$",
        )

    ax_r.plot(
        ridge["theta"],
        ridge["r_total"],
        "o-",
        ms=4,
        lw=1.5,
        color=COLOR_RT,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=r"$R_{\mathrm{total}}$",
    )
    ax_r.set_xlabel(r"$\phi_1$ ($^\circ$)")
    ax_r.set_ylabel("Reflectancia")
    ax_r.set_ylim(0, 1.02)
    ax_r.grid(True)
    ax_r.legend(loc="best", ncol=3)
    add_panel_label(ax_r, PANEL_LABELS[2])
    add_frequency_axis(ax_r, ridge["theta"], ridge["freq"])

    return fig, coeffs


def print_summary(ridge: dict[str, np.ndarray], coeffs: np.ndarray, mode: str) -> None:
    print(f"Par: {PAIR} | d1 = {D1_NM} nm | d2 = {D2_NM} nm | modo = {mode}")
    print("Cresta encontrada:")
    print(
        f"  phi = {ridge['theta'][0]:.0f} deg -> "
        f"{ridge['freq'][0]:.0f} cm^-1 "
        f"(CD = {ridge['cd'][0]:.3f}, R = {ridge['r_total'][0]:.3f})"
    )
    print(
        f"  phi = {ridge['theta'][-1]:.0f} deg -> "
        f"{ridge['freq'][-1]:.0f} cm^-1 "
        f"(CD = {ridge['cd'][-1]:.3f}, R = {ridge['r_total'][-1]:.3f})"
    )
    print(f"  Ajuste lineal: omega = {coeffs[0]:.3f}*phi + {coeffs[1]:.3f}")
    print(
        f"  CD: max = {np.max(ridge['cd']):.3f}, "
        f"min = {np.min(ridge['cd']):.3f}, "
        f"media = {np.mean(ridge['cd']):.3f}"
    )
    print(
        f"  R:  max = {np.max(ridge['r_total']):.3f}, "
        f"min = {np.min(ridge['r_total']):.3f}, "
        f"media = {np.mean(ridge['r_total']):.3f}"
    )
    print(
        f"  Rango espectral: {np.min(ridge['freq']):.0f}-"
        f"{np.max(ridge['freq']):.0f} cm^-1 "
        f"({1e4 / np.max(ridge['freq']):.1f}-"
        f"{1e4 / np.min(ridge['freq']):.1f} um)"
    )


def main() -> None:
    freqs = load_frequency_grid(ROOT_PATH)
    thetas = np.arange(0, 180 + THETA_STEP, THETA_STEP)
    mode = "TMM" if USE_TMM else "NN"

    print(f"Calculando water-plot: {len(thetas)} angulos x {len(freqs)} frecuencias")

    if USE_TMM:
        cd_matrix, r_matrix, rr_matrix, rl_matrix = compute_with_tmm(freqs, thetas)
    else:
        cd_matrix, r_matrix, rr_matrix, rl_matrix = compute_with_nn(ROOT_PATH, freqs, thetas)

    ridge = extract_ridge(freqs, thetas, cd_matrix, r_matrix, rr_matrix, rl_matrix)
    fig, coeffs = make_figure(freqs, thetas, cd_matrix, ridge, mode)
    print_summary(ridge, coeffs, mode)

    out_dir = Path(__file__).resolve().parent
    base_name = f"ridge_{PAIR}_d1{D1_NM}_d2{D2_NM}_{mode}"

    png_path = out_dir / f"{base_name}.png"
    pdf_path = out_dir / f"{base_name}.pdf"
    csv_path = save_ridge_table(out_dir, ridge, mode)

    fig.savefig(png_path)
    fig.savefig(pdf_path)

    arreglos = ROOT_PATH / "Arreglos en Gráficos"
    arreglos.mkdir(exist_ok=True)
    fig.savefig(arreglos / f"cresta_CD_{PAIR}_d1{D1_NM}_d2{D2_NM}.png")

    print(f"Figura PNG: {png_path}")
    print(f"Figura PDF: {pdf_path}")
    print(f"Tabla CSV:   {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
