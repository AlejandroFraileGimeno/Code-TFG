import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Permitividad hBN: modelo TO-LO
#
#  epsilon(w) = eps_inf * (w_LO^2 - w^2 - i gamma w)
#                        / (w_TO^2 - w^2 - i gamma w)
#
#  w, w_TO, w_LO y gamma están en cm^{-1}
# ============================================================

def epsilon_tolo(w, eps_inf, w_TO, w_LO, gamma):
    return eps_inf * (w_LO**2 - w**2 - 1j * gamma * w) / (
        w_TO**2 - w**2 - 1j * gamma * w
    )


# ============================================================
#  Parámetros hBN natural abundance
#
#      [ eps_perp      0          0     ]
#  ε = [    0       eps_perp      0     ]
#      [    0          0       eps_par  ]
#
#  UR: eps_perp < 0
#  LR: eps_par  < 0
# ============================================================

params = {
    "epsilon_perp": {
        "eps_inf": 4.90,
        "w_TO": 1360.0,
        "w_LO": 1614.0,
        "gamma": 7.0,
    },
    "epsilon_par": {
        "eps_inf": 2.95,
        "w_TO": 760.0,
        "w_LO": 825.0,
        "gamma": 3.0,
    },
}


# ============================================================
#  Ventanas espectrales
# ============================================================

w_LR = np.linspace(700, 850, 3000)
w_UR = np.linspace(1300, 1650, 4000)

eps_perp_LR = epsilon_tolo(w_LR, **params["epsilon_perp"])
eps_par_LR  = epsilon_tolo(w_LR, **params["epsilon_par"])

eps_perp_UR = epsilon_tolo(w_UR, **params["epsilon_perp"])
eps_par_UR  = epsilon_tolo(w_UR, **params["epsilon_par"])


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
#  Figura con eje x cortado
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    sharey=True,
    figsize=(7.6, 3.8),
    gridspec_kw={
        "width_ratios": [1.0, 2.15],
        "wspace": 0.06
    }
)

ylim = (-200, 200)


# ============================================================
#  Banda LR
# ============================================================

w_TO_par = params["epsilon_par"]["w_TO"]
w_LO_par = params["epsilon_par"]["w_LO"]

ax1.axvspan(
    w_TO_par, w_LO_par,
    facecolor="0.90",
    edgecolor="0.55",
    hatch="///",
    linewidth=0.0,
    zorder=0
)

ax1.plot(
    w_LR, np.real(eps_par_LR),
    color="black",
    linewidth=1.7,
    linestyle="-",
    label=r"$\mathrm{Re}(\varepsilon_{\parallel})$"
)

ax1.plot(
    w_LR, np.real(eps_perp_LR),
    color="black",
    linewidth=1.7,
    linestyle="--",
    label=r"$\mathrm{Re}(\varepsilon_{\perp})$"
)

ax1.text(
    0.5 * (w_TO_par + w_LO_par),
    0.78 * ylim[1],
    "LR",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)


# ============================================================
#  Banda UR
# ============================================================

w_TO_perp = params["epsilon_perp"]["w_TO"]
w_LO_perp = params["epsilon_perp"]["w_LO"]

ax2.axvspan(
    w_TO_perp, w_LO_perp,
    facecolor="0.90",
    edgecolor="0.55",
    hatch="///",
    linewidth=0.0,
    zorder=0
)

ax2.plot(
    w_UR, np.real(eps_par_UR),
    color="black",
    linewidth=1.7,
    linestyle="-",
    label=r"$\mathrm{Re}(\varepsilon_{\parallel})$"
)

ax2.plot(
    w_UR, np.real(eps_perp_UR),
    color="black",
    linewidth=1.7,
    linestyle="--",
    label=r"$\mathrm{Re}(\varepsilon_{\perp})$"
)

ax2.text(
    0.5 * (w_TO_perp + w_LO_perp),
    0.78 * ylim[1],
    "UR",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)


# ============================================================
#  Ajustes comunes
# ============================================================

for ax in (ax1, ax2):
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(*ylim)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()

ax1.set_xlim(700, 850)
ax2.set_xlim(1300, 1650)

# Ticks principales limpios
ax1.set_xticks([700, 760, 825])
ax2.set_xticks([1300, 1360, 1450, 1614])

ax1.set_ylabel(r"$\mathrm{Re}(\varepsilon)$")


# ============================================================
#  Corte del eje x
# ============================================================

ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

ax2.tick_params(labelleft=False, left=False)

d = 0.35
break_marker = [(-1, -d), (1, d)]

kwargs = dict(
    marker=break_marker,
    markersize=8,
    linestyle="none",
    color="black",
    mec="black",
    mew=1.0,
    clip_on=False
)

# Corte inferior
ax1.plot([1], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)

# Corte superior
ax1.plot([1], [1], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)


# ============================================================
#  Leyenda y etiqueta común del eje x
# ============================================================

ax2.legend(
    frameon=False,
    loc="lower right",
    handlelength=2.6
)

fig.subplots_adjust(bottom=0.22)

fig.text(
    0.52, 0.06,
    r"$\omega\ (\mathrm{cm}^{-1})$",
    ha="center",
    va="center"
)


# ============================================================
#  Guardar figura
# ============================================================

plt.savefig("hBN_Re_epsilon_Reststrahlen_BW.pdf", bbox_inches="tight")
plt.savefig("hBN_Re_epsilon_Reststrahlen_BW.png", dpi=400, bbox_inches="tight")

plt.show()
