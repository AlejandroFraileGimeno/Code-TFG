# -*- coding: utf-8 -*-
"""
Tabla de arquitectura de la red forward (surrogate) — estilo booktabs / LaTeX TFG
"""

from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        11,
    "text.usetex":      False,
})

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
headers = ["Capa", "Tipo", "Salida", "Parámetros", "Activación"]

rows = [
    ["Entrada",          "InputLayer", r"$(4,)$",    "—",       "—"],
    ["Densa 1",          "Dense",      r"$(256,)$",  "1 280",   "ReLU"],
    ["Densa 2",          "Dense",      r"$(512,)$",  "131 584", "ReLU"],
    ["Densa 3",          "Dense",      r"$(512,)$",  "262 656", "ReLU"],
    ["Densa 4",          "Dense",      r"$(256,)$",  "131 328", "ReLU"],
    [r"Salida $T_{xx}$", "Dense",      r"$(1000,)$", "257 000", "Sigmoide"],
]

# ---------------------------------------------------------------------------
# Figure — fixed size, axes fill the whole canvas
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 8.0, 3.8          # inches
MARGIN_L = 0.04                   # axes-coord left edge of table
MARGIN_R = 0.04                   # gap on the right
MARGIN_T = 0.12                   # top gap (for rules + breathing room)
MARGIN_B = 0.06                   # bottom gap

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------------------
# Vertical layout — evenly distribute rows in the available height
# ---------------------------------------------------------------------------
n_rows   = len(rows)
avail_h  = 1.0 - MARGIN_T - MARGIN_B     # 0.80 in axes coords
HEAD_H   = avail_h * 0.14                 # header ≈ 14 % of height
ROW_H    = (avail_h - HEAD_H) / n_rows   # equal row height

y_top    = 1.0 - MARGIN_T                # top of header
y_head   = y_top - HEAD_H               # bottom of header
y_bottom = y_head - n_rows * ROW_H      # bottom of last row

# ---------------------------------------------------------------------------
# Horizontal layout
# ---------------------------------------------------------------------------
# Proportional widths for each column (must sum ≤ 1 - MARGIN_L - MARGIN_R)
avail_w = 1.0 - MARGIN_L - MARGIN_R     # 0.92
weights  = [0.22, 0.24, 0.14, 0.19, 0.17]
COL_W    = [w / sum(weights) * avail_w for w in weights]

col_x = [MARGIN_L]
for w in COL_W[:-1]:
    col_x.append(col_x[-1] + w)
col_right = col_x[-1] + COL_W[-1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
INK       = "#0b0b0b"
INK_MUTED = "#52514e"

def hrule(y, lw):
    ax.plot([col_x[0], col_right], [y, y],
            color=INK, lw=lw, transform=ax.transAxes, clip_on=False, solid_capstyle="butt")

def cell_text(col, y_row, row_h, text, bold=False, align="center", size=10.5):
    xc = col_x[col] + COL_W[col] / 2 if align == "center" else col_x[col] + 0.011
    ax.text(xc, y_row + row_h / 2, text,
            ha=align, va="center",
            fontsize=size, color=INK,
            fontweight="bold" if bold else "normal",
            transform=ax.transAxes, clip_on=False)

# ---------------------------------------------------------------------------
# Toprule
# ---------------------------------------------------------------------------
hrule(y_top, lw=1.4)

# Header text
for ci, (h, w) in enumerate(zip(headers, COL_W)):
    cell_text(ci, y_head, HEAD_H, h, bold=True,
              align="left" if ci == 0 else "center")

# Midrule
hrule(y_head, lw=0.8)

# ---------------------------------------------------------------------------
# Data rows
# ---------------------------------------------------------------------------
for ri, row in enumerate(rows):
    y_row = y_head - (ri + 1) * ROW_H
    for ci, val in enumerate(row):
        cell_text(ci, y_row, ROW_H, val,
                  align="left" if ci == 0 else "center")

# Bottomrule
hrule(y_bottom, lw=1.4)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = Path(__file__).resolve().parent / "arquitectura_forward.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Guardado: {out}")
plt.show()
