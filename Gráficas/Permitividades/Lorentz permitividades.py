import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Modelo de Lorentz
#
# eps(w) = eps_inf + (eps_st - eps_inf) w0^2
#          / (w0^2 - w^2 - i gamma w)
#
# Aquí w está en unidades de 10^12 rad/s
# ============================================================

eps_inf = 10.0
eps_st  = 12.1

w0    = 100.0
gamma = 5.0

w = np.linspace(55, 145, 4000)

eps = eps_inf + (eps_st - eps_inf) * w0**2 / (
    w0**2 - w**2 - 1j * gamma * w
)

eps1 = np.real(eps)
eps2 = np.imag(eps)

# ============================================================
# Estilo
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 160,
})

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(4.2, 5.2),
    sharex=True,
    gridspec_kw={"hspace": 0.0}
)

# ============================================================
# Panel superior: parte real
# ============================================================

ax1.plot(w, eps1, color="black", lw=1.3)
ax1.axvline(w0, color="black", ls="--", lw=0.9)

# Líneas discontinuas parciales, como en la referencia
ax1.hlines(eps_st, xmin=50, xmax=88, colors="black", linestyles="--", linewidth=0.8)
ax1.hlines(eps_inf, xmin=112, xmax=142, colors="black", linestyles="--", linewidth=0.8)

# Etiquetas
ax1.text(63, eps_st - 2.7, r"$\varepsilon_{st}=12.1$", fontsize=12)
ax1.text(116, eps_inf + 1.0, r"$\varepsilon_{\infty}=10$", fontsize=12)

# Etiqueta explícita de la resonancia
ax1.text(w0 + 2.0, -8.5, r"$\omega_0$", fontsize=12)

ax1.set_ylabel(r"$\varepsilon'$")
ax1.set_xlim(55, 145)
ax1.set_ylim(-12, 34)

# ============================================================
# Panel inferior: parte imaginaria
# ============================================================

ax2.plot(w, eps2, color="black", lw=1.3)
ax2.axvline(w0, color="black", ls="--", lw=0.9)

# Volvemos a señalar explícitamente la resonancia
ax2.text(w0 + 2.0, 41, r"$\omega_0$", fontsize=12)

ax2.set_ylabel(r"$\varepsilon''$")
ax2.set_xlabel(r"$\omega\ (10^{12}\ \mathrm{rad/s})$")
ax2.set_ylim(0, 48)

# ============================================================
# Acabado final
# ============================================================

for ax in (ax1, ax2):
    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

plt.tight_layout()
plt.show()
