# -*- coding: utf-8 -*-
"""
Demo de diseno inverso: filtro stop-band con red tandem — T_xx  V2O5/MoO3/BaF2
================================================================================
Define un espectro objetivo stop-band (T~1 fuera, T~0 en la banda),
la inversa predice los parametros y se verifica con TMM.
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_PATH / "TMM"))

from generalized_transfer_matrix_method import (
    Air, BaF2, MoO3, V2O5, LayeredStructure, calculate_transmission,
)

# ============================================================
# CONFIG
# ============================================================
NUM_SEEDS   = 1
BAND_MIN    = 700.0
BAND_MAX    = 800.0
FLANK       = 30.0
T_STOP      = 0.0
T_PASS      = 1.0
# ============================================================

INVERSE_DIR = ROOT_PATH / "Models" / "T_xx" / "V2O5_MoO3_BaF2" / "Inverse"

scalers   = json.loads((INVERSE_DIR / "scalers.json").read_text())
param_min = np.array(scalers["param_min"], dtype=np.float32)
param_max = np.array(scalers["param_max"], dtype=np.float32)
N_FREQS   = scalers["n_freqs"]
FREQS     = np.linspace(scalers["freq_min"], scalers["freq_max"], N_FREQS)

print(f"Cargando {NUM_SEEDS} modelo(s) inverso(s)...")
inv_models = [
    tf.keras.models.load_model(INVERSE_DIR / f"Model_{i}seed" / "inverse.keras", compile=False)
    for i in range(1, NUM_SEEDS + 1)
]
print("Modelos cargados.\n")

def sigmoid_flank(f, center, width, invert=False):
    s = 1 / (1 + np.exp((f - center) / (width / 6)))
    return s if invert else 1 - s

T_target = (T_PASS * sigmoid_flank(FREQS, BAND_MIN, FLANK)
            + T_STOP * (sigmoid_flank(FREQS, BAND_MIN, FLANK, invert=True)
                        * sigmoid_flank(FREQS, BAND_MAX, FLANK))
            + T_PASS * sigmoid_flank(FREQS, BAND_MAX, FLANK, invert=True))
T_target = np.clip(T_target, 0, 1).astype(np.float32)

inp   = T_target.reshape(1, -1)
preds = [m.predict(inp, verbose=0)[0] for m in inv_models]
p_norm_mean = np.mean(preds, axis=0)
p_norm_std  = np.std(preds,  axis=0)

th1, th2, d1, d2 = p_norm_mean * (param_max - param_min) + param_min
unc_th1, unc_th2, unc_d1, unc_d2 = p_norm_std * (param_max - param_min)

print("Parametros predichos (media ensemble):")
print(f"  th1 = {th1:.1f} +/- {unc_th1:.1f} grados")
print(f"  th2 = {th2:.1f} +/- {unc_th2:.1f} grados")
print(f"  d1  = {d1:.0f} +/- {unc_d1:.0f} nm")
print(f"  d2  = {d2:.0f} +/- {unc_d2:.0f} nm")

print("\nVerificando con TMM...")
structure = LayeredStructure(
    superstrate=Air(), substrate=BaF2(),
    layers=[
        V2O5(d=d1 * 1e-9, phi=np.deg2rad(th1)),
        MoO3(d=d2 * 1e-9, phi=np.deg2rad(th2)),
    ],
)
T_tmm = np.array([float(calculate_transmission(f, 0, structure, basis="linear")[0])
                  for f in FREQS])

mae  = float(np.mean(np.abs(T_target - T_tmm)))
rmse = float(np.sqrt(np.mean((T_target - T_tmm) ** 2)))

mask_band = (FREQS >= BAND_MIN) & (FREQS <= BAND_MAX)
mask_pass = ~mask_band
T_stop_achieved = float(T_tmm[mask_band].mean())
T_pass_achieved = float(T_tmm[mask_pass].mean())
rejection_db    = -10 * np.log10(max(T_stop_achieved, 1e-6))

print(f"\nResultados TMM:")
print(f"  T media en banda suprimida ({BAND_MIN:.0f}-{BAND_MAX:.0f} cm-1): "
      f"{T_stop_achieved:.4f}  ({rejection_db:.1f} dB de rechazo)")
print(f"  T media fuera de la banda:                         {T_pass_achieved:.4f}")
print(f"  MAE  (objetivo vs TMM): {mae:.5f}")
print(f"  RMSE (objetivo vs TMM): {rmse:.5f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(FREQS, T_target, "k-",  lw=2, label="Objetivo stop-band")
ax1.plot(FREQS, T_tmm,    "r--", lw=2, label="TMM (params inversa)")
ax1.axvspan(BAND_MIN, BAND_MAX, alpha=0.1, color="red", label="Banda suprimida")
ax1.axhline(T_stop_achieved, color="red",   ls=":", lw=1, alpha=0.7)
ax1.axhline(T_pass_achieved, color="green", ls=":", lw=1, alpha=0.7)
ax1.set_xlabel("Numero de onda (cm-1)", fontsize=12)
ax1.set_ylabel("T_xx", fontsize=12)
ax1.set_ylim(-0.05, 1.1)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title(
    f"Stop-band {BAND_MIN:.0f}-{BAND_MAX:.0f} cm-1\n"
    f"th1={th1:.0f}  th2={th2:.0f}  d1={d1:.0f}  d2={d2:.0f} nm",
    fontsize=11,
)
ax1.annotate(
    f"Rechazo: {rejection_db:.1f} dB\nT_stop={T_stop_achieved:.3f}",
    xy=((BAND_MIN + BAND_MAX) / 2, T_stop_achieved),
    xytext=((BAND_MIN + BAND_MAX) / 2 + 80, 0.3),
    fontsize=9, color="red",
    arrowprops=dict(arrowstyle="->", color="red"),
)

ax2 = axes[1]
ax2.plot(FREQS, T_target, "k-", lw=2, label="Objetivo", zorder=5)
colors = ["tab:blue", "tab:orange", "tab:green"]
for i, (p_norm, col) in enumerate(zip(preds, colors)):
    th1_i, th2_i, d1_i, d2_i = p_norm * (param_max - param_min) + param_min
    s_i = LayeredStructure(
        superstrate=Air(), substrate=BaF2(),
        layers=[V2O5(d=d1_i*1e-9, phi=np.deg2rad(th1_i)),
                MoO3(d=d2_i*1e-9, phi=np.deg2rad(th2_i))],
    )
    T_i = np.array([float(calculate_transmission(f, 0, s_i, basis="linear")[0]) for f in FREQS])
    ax2.plot(FREQS, T_i, "--", color=col, lw=1.2, alpha=0.8,
             label=f"Modelo {i+1}: th1={th1_i:.0f} th2={th2_i:.0f} d1={d1_i:.0f} d2={d2_i:.0f}")

ax2.axvspan(BAND_MIN, BAND_MAX, alpha=0.1, color="red")
ax2.set_xlabel("Numero de onda (cm-1)", fontsize=12)
ax2.set_ylabel("T_xx", fontsize=12)
ax2.set_ylim(-0.05, 1.1)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_title("Respuesta individual de cada modelo del ensemble", fontsize=11)

fig.tight_layout()
out = Path(__file__).parent / f"demo_stopband_{BAND_MIN:.0f}_{BAND_MAX:.0f}.png"
fig.savefig(out, dpi=150)
print(f"\nGuardado: {out.name}")
plt.show()