# -*- coding: utf-8 -*-
"""
Generador de bicapas aleatorias.
Ejecuta indefinidamente hasta encontrar un caso fisicamente imposible:
  - Tij > 1  o  Rij > 1

Al encontrarlo, plotea R y T y guarda la figura en CASOS_ANOMALOS/.
Pulsa Ctrl+C para parar manualmente.
"""

import csv
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH  = Path(__file__).resolve().parents[2]
CASOS_DIR  = Path(__file__).parent / "CASOS_ANOMALOS"
CASOS_DIR.mkdir(exist_ok=True)
CSV_LOG    = CASOS_DIR / "log.csv"

sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2,
    MoO3, V2O5, MgTeMoO6, hBN,
    LayeredStructure,
    calculate_reflection,
    calculate_transmission,
)

MATERIALES = [MoO3, V2O5, MgTeMoO6, hBN]

N_FREQS       = 200
FREQ_MIN      = 400
FREQ_MAX      = 1400
TOL           = 1e-6
N_CASOS_MAX   = 100     # Numero de casos anomalos a encontrar antes de parar


def random_structure():
    mat1  = random.choice(MATERIALES)
    mat2  = random.choice(MATERIALES)
    sust  = BaF2
    d1    = random.uniform(200e-9, 2000e-9)
    d2    = random.uniform(200e-9, 2000e-9)
    phi1  = random.uniform(0, 2 * np.pi)
    phi2  = random.uniform(0, 2 * np.pi)
    alpha = 0.0
    return mat1, mat2, sust, d1, d2, phi1, phi2, alpha


def check_impossible(R, T):
    msgs = []
    for j, name in enumerate(["xx", "yy", "xy", "yx"]):
        if np.any(R[:, j] > 1 + TOL):
            msgs.append(f"R{name} > 1  (max={R[:, j].max():.6f})")
        if np.any(T[:, j] > 1 + TOL):
            msgs.append(f"T{name} > 1  (max={T[:, j].max():.6f})")
    return msgs


# Estilo TFG
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           12,
    "axes.labelsize":      13,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.linewidth":      0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "legend.fontsize":     11,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#c3c2b7",
    "axes.grid":           True,
    "grid.linewidth":      0.5,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
})

# Paleta fija (orden validado): xx azul, yy aqua, xy amarillo, yx verde
_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]


def plot_caso(omega, R, T, mat1, mat2, sust, d1, d2, phi1, phi2, alpha, count, msgs):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(omega, R[:, 0], color=_COLORS[0], lw=1.6, label=r"$R_{xx}$")
    axs[0].plot(omega, R[:, 1], color=_COLORS[1], lw=1.6, label=r"$R_{yy}$")
    axs[0].plot(omega, R[:, 2], color=_COLORS[2], lw=1.4, linestyle="--", label=r"$R_{xy}$")
    axs[0].plot(omega, R[:, 3], color=_COLORS[3], lw=1.4, linestyle="--", label=r"$R_{yx}$")
    axs[0].axhline(1, color="#0b0b0b", lw=0.8, linestyle=":")
    axs[0].set_ylabel("Reflectancia")
    axs[0].legend()

    axs[1].plot(omega, T[:, 0], color=_COLORS[0], lw=1.6, label=r"$T_{xx}$")
    axs[1].plot(omega, T[:, 1], color=_COLORS[1], lw=1.6, label=r"$T_{yy}$")
    axs[1].plot(omega, T[:, 2], color=_COLORS[2], lw=1.4, linestyle="--", label=r"$T_{xy}$")
    axs[1].plot(omega, T[:, 3], color=_COLORS[3], lw=1.4, linestyle="--", label=r"$T_{yx}$")
    axs[1].axhline(1, color="#0b0b0b", lw=0.8, linestyle=":")
    axs[1].set_ylabel("Transmitancia")
    axs[1].set_xlabel(r"$\omega$ (cm$^{-1}$)")
    axs[1].legend()

    title = (
        f"CASO ANOMALO #{count}\n"
        f"Air / {mat1.__name__}({d1*1e9:.0f}nm, {np.degrees(phi1):.1f}deg)"
        f" / {mat2.__name__}({d2*1e9:.0f}nm, {np.degrees(phi2):.1f}deg)"
        f" / {sust.__name__}   alpha={np.degrees(alpha):.1f}deg\n"
        + "  |  ".join(msgs)
    )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()

    out = CASOS_DIR / f"caso_{count:05d}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out}")


omega = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

print("Iniciando experimento — Ctrl+C para parar")
print(f"  {N_FREQS} frecuencias | rango {FREQ_MIN}-{FREQ_MAX} cm^-1\n")

CSV_HEADER = ["estructura", "mat1", "mat2", "d1_nm", "d2_nm", "phi1_deg", "phi2_deg", "violaciones", "max_val"]
csv_nuevo = not CSV_LOG.exists()
csv_file  = open(CSV_LOG, "a", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
if csv_nuevo:
    csv_writer.writeheader()

count       = 0
casos_found = 0
try:
    while casos_found < N_CASOS_MAX:
        mat1, mat2, sust, d1, d2, phi1, phi2, alpha = random_structure()

        structure = LayeredStructure(
            superstrate=Air(),
            substrate=sust(),
            layers=[
                mat1(d=d1, phi=phi1),
                mat2(d=d2, phi=phi2),
            ],
        )

        T = np.zeros((N_FREQS, 4))
        R = np.zeros((N_FREQS, 4))

        ok = True
        for i in range(N_FREQS):
            try:
                t = calculate_transmission(omega[i], alpha, structure)
                r = calculate_reflection(omega[i], alpha, structure)
            except Exception as e:
                print(f"[#{count+1}] EXCEPCION: {e}")
                ok = False
                break
            for j in range(4):
                T[i, j] = t[j]
                R[i, j] = r[j]

        if not ok:
            count += 1
            continue

        count += 1
        if count % 100 == 0:
            print(f"  {count} estructuras probadas, {casos_found}/{N_CASOS_MAX} casos anomalos...")

        msgs = check_impossible(R, T)
        if msgs:
            print(f"\n*** CASO ANOMALO #{count} ***")
            print(f"  Capa 1 : {mat1.__name__}  d={d1*1e9:.1f} nm  phi={np.degrees(phi1):.1f} deg")
            print(f"  Capa 2 : {mat2.__name__}  d={d2*1e9:.1f} nm  phi={np.degrees(phi2):.1f} deg")
            print(f"  Sustrato: {sust.__name__}")
            print(f"  Alpha  : {np.degrees(alpha):.2f} deg")
            print(f"  Violaciones:")
            for m in msgs:
                print(f"    - {m}")
            casos_found += 1
            plot_caso(omega, R, T, mat1, mat2, sust, d1, d2, phi1, phi2, alpha, count, msgs)
            max_val = max(R.max(), T.max())
            csv_writer.writerow({
                "estructura": count,
                "mat1":       mat1.__name__,
                "mat2":       mat2.__name__,
                "d1_nm":      f"{d1*1e9:.1f}",
                "d2_nm":      f"{d2*1e9:.1f}",
                "phi1_deg":   f"{np.degrees(phi1):.2f}",
                "phi2_deg":   f"{np.degrees(phi2):.2f}",
                "violaciones": " | ".join(msgs),
                "max_val":    f"{max_val:.6f}",
            })
            csv_file.flush()
            print(f"  ({casos_found}/{N_CASOS_MAX} casos encontrados)\n")

except KeyboardInterrupt:
    print(f"\nParado manualmente tras {count} estructuras.")
finally:
    csv_file.close()

print(f"\nTotal estructuras evaluadas: {count}  |  Casos anomalos encontrados: {casos_found}")
