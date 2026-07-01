import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Permitividad α-MoO3: modelo TO-LO multi-oscilador
#
#  epsilon_j(w) = eps_inf_j * Π_n
#      (w_LO,n^2 - w^2 - i gamma_n w)
#      --------------------------------
#      (w_TO,n^2 - w^2 - i gamma_n w)
#
#  w, w_TO, w_LO y gamma están en cm^{-1}
# ============================================================

def epsilon_tolo_multi(w, eps_inf, modes):
    eps = eps_inf * np.ones_like(w, dtype=complex)

    for mode in modes:
        w_TO = mode["w_TO"]
        w_LO = mode["w_LO"]
        gamma = mode["gamma"]

        eps *= (w_LO**2 - w**2 - 1j * gamma * w) / (
            w_TO**2 - w**2 - 1j * gamma * w
        )

    return eps


# ============================================================
#  Parámetros α-MoO3
#
#      [ eps_x      0        0   ]
#  ε = [   0      eps_y      0   ]
#      [   0        0      eps_z ]
# ============================================================

params = {
    "epsilon_x": {
        "eps_inf": 5.78,
        "modes": [
            {"w_TO": 506.7, "w_LO": 534.3, "gamma": 49.1},
            {"w_TO": 821.4, "w_LO": 963.0, "gamma": 6.0},
            {"w_TO": 998.7, "w_LO": 999.2, "gamma": 0.35},
        ],
    },
    "epsilon_y": {
        "eps_inf": 6.07,
        "modes": [
            {"w_TO": 544.6, "w_LO": 850.1, "gamma": 9.5},
        ],
    },
    "epsilon_z": {
        "eps_inf": 4.47,
        "modes": [
            {"w_TO": 956.7, "w_LO": 1006.9, "gamma": 1.5},
        ],
    },
}


# ============================================================
#  Ventana espectral
# ============================================================

w = np.linspace(450, 1050, 6000)

eps_x = epsilon_tolo_multi(w, **params["epsilon_x"])
eps_y = epsilon_tolo_multi(w, **params["epsilon_y"])
eps_z = epsilon_tolo_multi(w, **params["epsilon_z"])


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
#  RB1: asociada principalmente a epsilon_y
#  RB2: asociada principalmente a epsilon_x
#  RB3: asociada principalmente a epsilon_z
#
#  Las zonas más oscuras indican solapamiento entre bandas
# ============================================================

RB1 = (544.6, 850.1)
RB2 = (821.4, 963.0)
RB3 = (956.7, 1006.9)

overlap_12 = (max(RB1[0], RB2[0]), min(RB1[1], RB2[1]))
overlap_23 = (max(RB2[0], RB3[0]), min(RB2[1], RB3[1]))

# Bandas individuales
for RB in [RB1, RB2, RB3]:
    ax.axvspan(
        RB[0], RB[1],
        facecolor="0.92",
        edgecolor="0.65",
        hatch="///",
        linewidth=0.0,
        zorder=0
    )

# Zonas de solapamiento
for overlap in [overlap_12, overlap_23]:
    ax.axvspan(
        overlap[0], overlap[1],
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
    0.5 * (RB1[0] + RB1[1]),
    0.82 * ylim[1],
    r"RB$_1$",
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
    0.5 * (RB3[0] + RB3[1]),
    0.82 * ylim[1],
    r"RB$_3$",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)


# ============================================================
#  Ajustes de ejes
# ============================================================

ax.axhline(0, color="black", linewidth=0.8, zorder=1.5)

ax.set_xlim(450, 1050)
ax.set_ylim(*ylim)

ax.set_xlabel(r"$\omega\ (\mathrm{cm}^{-1})$")
ax.set_ylabel(r"$\mathrm{Re}(\varepsilon)$")

# Ticks uniformes en el eje x
ax.set_xticks([500, 600, 700, 800, 900, 1000])

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

plt.savefig("MoO3_Re_epsilon_Reststrahlen_BW.pdf", bbox_inches="tight")
plt.savefig("MoO3_Re_epsilon_Reststrahlen_BW.png", dpi=400, bbox_inches="tight")

plt.show()
