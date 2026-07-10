# -*- coding: utf-8 -*-
"""
Transmitancia vs numero de onda para laminas individuales.
Genera los dos paneles de la figura de transmitancia:
Air / material(d, phi) / BaF2.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2,
    MoO3, V2O5,
    LayeredStructure,
    calculate_transmission,
)


CASES = [
    {
        "material": V2O5,
        "filename": "V2O5 2000 nm",
        "out_name": "transmitancia_V2O5_2000nm",
        "title": r"$\alpha\mathrm{-V_2O_5}\ (d = 2000\,\mathrm{nm})$",
        "thickness": 2000e-9,
        "freq_min": 500,
        "freq_max": 1100,
        "component": "ss",
    },
    {
        "material": MoO3,
        "filename": "MoO3 100 nm",
        "out_name": "transmitancia_MoO3_100nm",
        "title": r"$\alpha\mathrm{-MoO_3}\ (d = 100\,\mathrm{nm})$",
        "thickness": 100e-9,
        "freq_min": 400,
        "freq_max": 1100,
        "component": "ss",
    },
]

SUBSTRATE = BaF2
PHI = 0.0
ALPHA = 0.0
N_FREQS = 1000

_T_INDEX = {"pp": 0, "ss": 1, "ps": 2, "sp": 3}
_T_LABEL = {"pp": "xx", "ss": "yy", "ps": "xy", "sp": "yx"}


plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           17,
    "axes.labelsize":      19,
    "axes.titlesize":      18,
    "xtick.labelsize":     16,
    "ytick.labelsize":     16,
    "axes.linewidth":      1.1,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     15,
    "legend.framealpha":   0.95,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.6,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})


def transmission_spectrum(case):
    component = case["component"]
    if component not in _T_INDEX:
        raise ValueError(f"Componente desconocida: {component}")

    omega = np.linspace(case["freq_min"], case["freq_max"], N_FREQS)
    structure = LayeredStructure(
        superstrate=Air(),
        substrate=SUBSTRATE(),
        layers=[case["material"](d=case["thickness"], phi=PHI)],
    )

    t_index = _T_INDEX[component]
    transmittance = np.array([
        float(calculate_transmission(freq, ALPHA, structure, basis="linear")[t_index])
        for freq in omega
    ])
    return omega, transmittance


def save_case(case):
    omega, transmittance = transmission_spectrum(case)
    component_label = _T_LABEL[case["component"]]

    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    ax.plot(
        omega,
        transmittance,
        color="#0b0b0b",
        linewidth=2.6,
        label=rf"$T_{{{component_label}}}$",
    )

    ax.set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$")
    ax.set_ylabel("Transmitancia")
    ax.set_title(case["title"], pad=8)
    ax.set_xlim(case["freq_min"], case["freq_max"])
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="major", width=1.1, length=5)
    ax.tick_params(which="minor", width=0.9, length=3)

    fig.tight_layout()

    local_out = Path(__file__).resolve().parent
    arreglo_out = ROOT_PATH / "Arreglos en Gráficos"
    arreglo_out.mkdir(exist_ok=True)

    fig.savefig(local_out / f"{case['filename']}.png", dpi=300, bbox_inches="tight")
    fig.savefig(local_out / f"{case['filename']}.pdf", bbox_inches="tight")
    fig.savefig(arreglo_out / f"{case['out_name']}.png", dpi=300, bbox_inches="tight")
    fig.savefig(arreglo_out / f"{case['out_name']}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Guardado: {case['filename']} y {case['out_name']}")


for case in CASES:
    save_case(case)
