import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Permitividad α-V2O5: modelo TO-LO
#
#  epsilon_j(w) = eps_inf_j *
#      (w_LO^2 - w^2 - i Gamma w)
#      ---------------------------
#      (w_TO^2 - w^2 - i Gamma w)
#
#  w, w_TO, w_LO y Gamma están en cm^{-1}
# ============================================================

def epsilon_tolo(w, eps_inf, w_TO, w_LO, gamma):
    return eps_inf * (w_LO**2 - w**2 - 1j * gamma * w) / (
        w_TO**2 - w**2 - 1j * gamma * w
    )


# ============================================================
#  Parámetros α-V2O5
#
#      [ eps_x      0        0   ]
#  ε = [   0      eps_y      0   ]
#      [   0        0      eps_z ]
#
#  Direcciones:
#  eps_x -> [100]
#  eps_y -> [001]
#  eps_z -> [010]
#
#  Datos tomados de la Tabla S2 de Taboada-Gutiérrez et al.
# ============================================================

params = {
    "epsilon_x": {
        "eps_inf": 6.559,
        "w_TO": 770.0,
        "w_LO": 944.3,
        "gamma": 8.1,
    },
    "epsilon_y": {
        "eps_inf": 6.142,
        "w_TO": 474.4,
        "w_LO": 815.6,
        "gamma": 9.6,
    },
    "epsilon_z": {
        "eps_inf": 3.899,
        "w_TO": 1004.4,
        "w_LO": 1073.4,
        "gamma": 0.56,
    },
}


# ============================================================
#  Ventana espectral
# ============================================================

w = np.linspace(400, 1125, 7000)

eps_x = epsilon_tolo(w, **params["epsilon_x"])
eps_y = epsilon_tolo(w, **params["epsilon_y"])
eps_z = epsilon_tolo(w, **params["epsilon_z"])


# ============================================================
#  Estilo general
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


# ============================================================
#  Figura
# ============================================================

fig, ax = plt.subplots(figsize=(7.6, 3.8))

ylim = (-300, 300)


# ============================================================
#  Bandas de Reststrahlen
#
#  Según la notación del paper:
#
#  RB1: asociada principalmente a epsilon_z
#  RB2: asociada principalmente a epsilon_x
#  RB3: asociada principalmente a epsilon_y
#
#  Las zonas más oscuras indican solapamiento entre bandas
# ============================================================

RB1 = (params["epsilon_z"]["w_TO"], params["epsilon_z"]["w_LO"])  # epsilon_z
RB2 = (params["epsilon_x"]["w_TO"], params["epsilon_x"]["w_LO"])  # epsilon_x
RB3 = (params["epsilon_y"]["w_TO"], params["epsilon_y"]["w_LO"])  # epsilon_y

# En α-V2O5 hay solapamiento entre RB2 y RB3
overlap_23 = (max(RB2[0], RB3[0]), min(RB2[1], RB3[1]))

# Bandas individuales
for RB in [RB3, RB2, RB1]:
    ax.axvspan(
        RB[0], RB[1],
        facecolor="0.92",
        edgecolor="0.65",
        hatch="///",
        linewidth=0.0,
        zorder=0
    )

# Zona de solapamiento
ax.axvspan(
    overlap_23[0], overlap_23[1],
    facecolor="0.70",
    edgecolor="0.35",
    hatch="xxx",
    linewidth=0.0,
    zorder=0.2
)


# ============================================================
#  Curvas Re(epsilon)
# ============================================================

ax.plot(
    w, np.real(eps_x),
    color="black",
    linewidth=1.7,
    linestyle="-",
    label=r"$\mathrm{Re}(\varepsilon_x)$",
    zorder=2
)

ax.plot(
    w, np.real(eps_y),
    color="black",
    linewidth=1.7,
    linestyle="--",
    label=r"$\mathrm{Re}(\varepsilon_y)$",
    zorder=2
)

ax.plot(
    w, np.real(eps_z),
    color="black",
    linewidth=1.7,
    linestyle=":",
    label=r"$\mathrm{Re}(\varepsilon_z)$",
    zorder=2
)


# ============================================================
#  Etiquetas RB
# ============================================================

ax.text(
    0.5 * (RB3[0] + RB3[1]),
    0.82 * ylim[1],
    r"RB$_3$",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    0.5 * (RB2[0] + RB2[1]),
    0.82 * ylim[1],
    r"RB$_2$",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    0.5 * (RB1[0] + RB1[1]),
    0.82 * ylim[1],
    r"RB$_1$",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)


# ============================================================
#  Ajustes de ejes
# ============================================================

ax.axhline(0, color="black", linewidth=0.8, zorder=1.5)

ax.set_xlim(400, 1125)
ax.set_ylim(*ylim)

ax.set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$")
ax.set_ylabel(r"$\mathrm{Re}(\varepsilon)$")

# Ticks uniformes
ax.set_xticks([400, 500, 600, 700, 800, 900, 1000, 1100])

ax.minorticks_on()

ax.tick_params(which="both", direction="in", top=True, right=True)


# ============================================================
#  Leyenda
# ============================================================

ax.legend(
    frameon=False,
    loc="lower right",
    handlelength=2.6
)


# ============================================================
#  Guardar figura
# ============================================================

plt.savefig("V2O5_Re_epsilon_Reststrahlen_BW.pdf", bbox_inches="tight")
plt.savefig("V2O5_Re_epsilon_Reststrahlen_BW.png", dpi=400, bbox_inches="tight")

plt.show()
