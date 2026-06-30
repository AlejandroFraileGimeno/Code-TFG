# -*- coding: utf-8 -*-
"""
Analisis de anomalias (R>1 o T>1) en bicapas aleatorias.
Corre N_EVAL estructuras aleatorias, anota las defectuosas en CSV
y al final muestra estadisticas y graficas.
"""

import csv
import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).resolve().parents[2]
OUT_DIR   = Path(__file__).parent / "CASOS_ANOMALOS"
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH  = OUT_DIR / "analisis.csv"

sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2,
    MoO3, V2O5, MgTeMoO6, hBN,
    LayeredStructure,
    calculate_reflection,
    calculate_transmission,
)

MATERIALES = [MoO3, V2O5, MgTeMoO6, hBN]

N_EVAL   = 5000
N_FREQS  = 100
FREQ_MIN = 400
FREQ_MAX = 1400
TOL      = 1e-6
alpha    = 0.0

omega = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)


def evaluar(mat1, mat2, phi1, phi2, d1, d2):
    structure = LayeredStructure(
        superstrate=Air(),
        substrate=BaF2(),
        layers=[
            mat1(d=d1, phi=phi1),
            mat2(d=d2, phi=phi2),
        ],
    )
    T = np.zeros((N_FREQS, 4))
    R = np.zeros((N_FREQS, 4))
    for i in range(N_FREQS):
        try:
            t = calculate_transmission(omega[i], alpha, structure)
            r = calculate_reflection(omega[i], alpha, structure)
        except Exception:
            return None, None
        for j in range(4):
            T[i, j] = t[j]
            R[i, j] = r[j]
    return R, T


def componentes_violadas(R, T):
    msgs = []
    for j, name in enumerate(["xx", "yy", "yx", "xy"]):
        if np.any(R[:, j] > 1 + TOL):
            msgs.append(f"R{name}")
        if np.any(T[:, j] > 1 + TOL):
            msgs.append(f"T{name}")
    return msgs


print(f"Corriendo {N_EVAL} estructuras aleatorias...")

registros = []

for n in range(N_EVAL):
    mat1 = random.choice(MATERIALES)
    mat2 = random.choice(MATERIALES)
    phi1 = random.uniform(0, 2 * np.pi)
    phi2 = random.uniform(0, 2 * np.pi)
    d1   = random.uniform(200e-9, 2000e-9)
    d2   = random.uniform(200e-9, 2000e-9)

    R, T = evaluar(mat1, mat2, phi1, phi2, d1, d2)
    if R is None:
        continue

    viols = componentes_violadas(R, T)
    if viols:
        registros.append({
            "mat1":    mat1.__name__,
            "mat2":    mat2.__name__,
            "phi1":    round(np.degrees(phi1), 2),
            "phi2":    round(np.degrees(phi2), 2),
            "d1_nm":   round(d1 * 1e9, 1),
            "d2_nm":   round(d2 * 1e9, 1),
            "max_val": round(max(R.max(), T.max()), 6),
            "viols":   " ".join(viols),
        })

    if (n + 1) % 500 == 0:
        print(f"  {n+1}/{N_EVAL}  anomalos: {len(registros)}")

print(f"\nTotal evaluadas: {N_EVAL}  |  Anomalas: {len(registros)}  ({100*len(registros)/N_EVAL:.1f}%)\n")

# Guardar CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["mat1","mat2","phi1","phi2","d1_nm","d2_nm","max_val","viols"])
    writer.writeheader()
    writer.writerows(registros)
print(f"CSV guardado: {CSV_PATH}\n")

if not registros:
    print("Sin anomalias. Prueba a aumentar N_EVAL.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Estadisticas
# ---------------------------------------------------------------------------
pares   = Counter(f"{r['mat1']}/{r['mat2']}" for r in registros)
viols_c = Counter(v for r in registros for v in r["viols"].split())
max_val = [r["max_val"] for r in registros]

print("Pares mas frecuentes:")
for par, cnt in pares.most_common():
    print(f"  {par:28s}  {cnt:4d}  ({100*cnt/len(registros):.1f}%)")

print("\nComponentes violadas:")
for comp, cnt in viols_c.most_common():
    print(f"  {comp}  {cnt}")

# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Barras por par
pares_labels = [p for p, _ in pares.most_common()]
pares_counts = [pares[p] for p in pares_labels]
axes[0].barh(pares_labels[::-1], pares_counts[::-1], color="tab:red")
axes[0].set_xlabel("Nº anomalias")
axes[0].set_title("Anomalias por par de materiales")
axes[0].grid(axis="x", alpha=0.3)

# 2. Scatter phi1 vs phi2
phi1s = [r["phi1"] for r in registros]
phi2s = [r["phi2"] for r in registros]
axes[1].scatter(phi1s, phi2s, c=max_val, cmap="Reds", s=10, vmin=1)
axes[1].set_xlabel("phi1 (deg)"); axes[1].set_ylabel("phi2 (deg)")
axes[1].set_title("phi1 vs phi2 en anomalias")

# 3. Scatter d1 vs d2
d1s = [r["d1_nm"] for r in registros]
d2s = [r["d2_nm"] for r in registros]
sc = axes[2].scatter(d1s, d2s, c=max_val, cmap="Reds", s=10, vmin=1)
axes[2].set_xlabel("d1 (nm)"); axes[2].set_ylabel("d2 (nm)")
axes[2].set_title("d1 vs d2 en anomalias")
fig.colorbar(sc, ax=axes[2], label="max(R,T)")

fig.tight_layout()
out = OUT_DIR / "analisis_anomalias.png"
fig.savefig(out, dpi=150)
print(f"\nGrafica guardada: {out}")
plt.show()
