import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURACIÓN ---
plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams.update({
    'font.family':         'serif',
    'mathtext.fontset':    'cm',
    'font.size':           14,
    'axes.labelsize':      16,
    'legend.fontsize':     13,
    'xtick.labelsize':     13,
    'ytick.labelsize':     13,
    'lines.markersize':    12,
    'axes.linewidth':      0.9,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'ytick.minor.visible': True,
})

# --- 2. DATOS ---
datos_materiales = [
    (r'$\alpha$-MoO$_3$',    [506.7, 810],                        [545.6]),
    (r'$\alpha$-V$_2$O$_5$', [72.4, 261, 303, 411, 767.5, 980.5], [212.0, 284.0, 506.5]),
    (r'hBN',                  [1372],                              [1372]),
    (r'MgTeMoO$_6$',          [698.47, 749.47, 895.50],            [684.54, 749.08, 902.98]),
]

colores = {
    r'$\alpha$-MoO$_3$':    '#2166ac',
    r'$\alpha$-V$_2$O$_5$': '#d6604d',
    r'hBN':                  '#1a9641',
    r'MgTeMoO$_6$':          '#762a83',
}

# --- 3. GRAFICAR ---
fig, ax = plt.subplots()

for nombre, tox, toy in datos_materiales:
    color = colores[nombre]
    ax.plot(np.ones(len(tox)), tox, 'o', color=color, label=nombre, linestyle='', markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(np.ones(len(toy)) * 2, toy, 'o', color=color, linestyle='', markeredgecolor='black', markeredgewidth=0.5)

# --- 4. AJUSTES ---
ax.set_xlim(0, 3)
ax.set_xticks([1, 2])
ax.set_xticklabels(['TO$_X$ Phonon', 'TO$_Y$ Phonon'])

# --- AQUÍ ESTÁ EL CAMBIO ---
# pad=15 mueve las etiquetas hacia abajo (sepáralas más subiendo el número)
ax.tick_params(axis='x', which='major', pad=15) 

ax.set_ylabel(r'$\omega$ (cm$^{-1}$)')
ax.set_ylim(400, 1400)

# --- 5. LEYENDA ---
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

ax.legend(by_label.values(), by_label.keys(), 
          loc='upper left',          
          bbox_to_anchor=(0.5, 0.9), # He mantenido la posición que pusiste en tu código
          frameon=True, 
          edgecolor='black')

plt.tight_layout()
plt.show()