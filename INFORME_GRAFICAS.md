# Informe — Unificación estética de las gráficas del TFG

**Fecha:** 04/07/2026
**Alcance:** 102 scripts modificados (solo estética). **Cero cambios en cálculos, datos, series mostradas, rutas o nombres de ficheros de salida.**

---

## 1. Diagnóstico

El repositorio tenía dos "generaciones" de gráficas conviviendo:

- **Buenas (referencia):** `Tandem\T_xx\plot_forward_single.py`, `plot_tandem_single.py`, `plot_training_history.py` y toda la carpeta `Gráficas\Permitividades\` — estilo publicación (serif + Computer Modern, ticks hacia dentro, colores sobrios).
- **Desastre (el resto, ~100 scripts):** estilo matplotlib por defecto, y en concreto:
  - **Nubes de scatter** azul/rojo puro (`c="blue"`, `c="red"`, `s=5`) para espectros de 1000 puntos que son líneas continuas (todos los `plot_cd.py`, `plot_r_total.py`, `TMM_NN.py`, `de_optimizer.py`).
  - **Colormap `RdYlGn`** en los heatmaps de calidad R² de los 25 demos de Tandem — un rojo-verde ilegible para daltónicos y de percepción no uniforme.
  - **Etiquetas ASCII rotas**: "Numero de onda (cm-1)", "theta (deg)", "sigma (cm-1)", "lambda target", sin acentos ni LaTeX, mezcladas al azar con versiones Unicode en otras copias.
  - **Etiqueta físicamente incorrecta**: los `plot_sweep.py` llamaban "λ target (cm⁻¹)" a un eje que está en número de onda.
  - **Colores `tab:` cíclicos** sin significado fijo (el mismo dato cambiaba de color entre scripts).
  - Sin tipografía definida, `dpi=150` o por defecto, sin minor ticks, grids ruidosos e inconsistentes.

## 2. Sistema de estilo aplicado ("Estilo TFG")

Se tomó como base el estilo ya existente en los `plot_*_single.py` y se extendió a todo:

**Tipografía y chrome** (bloque `rcParams` insertado en cada script):
serif + mathtext "cm" (Computer Modern, como el LaTeX del TFG) · ticks hacia dentro en los 4 lados con minor ticks · grid discontinuo sutil (lw 0.5, alpha 0.35) · leyenda con marco `#c3c2b7` · `savefig` a **dpi 200** con `bbox_inches="tight"`.

**Paleta por rol** (el color sigue al significado, nunca al orden de dibujo — validada para visión con daltonismo):

| Rol | Color | Trazo |
|---|---|---|
| Verdad de campo (TMM / objetivo) | `#0b0b0b` negro | sólido, lw 2.0 |
| Predicción NN / surrogate / reconstruido | `#2a78d6` azul | discontinuo, lw 1.6 |
| Segunda predicción (forward NN) / serie acento (R_total, R_r) | `#e34948` rojo | discontinuo, lw 1.4 |
| Serie neutra / medias / referencias | `#898781` gris | — |
| Ventanas de evaluación / bandas | `#e1e0d9` gris claro | relleno |
| 4 componentes T/R (xx, yy, xy, yx) | `#2a78d6`, `#1baf7a`, `#eda100`, `#008300` | orden fijo; cross-pol discontinuo |

**Heatmaps:** `inferno` para |CD| (se mantiene), `viridis` para transmitancia y para los mapas de calidad R² (sustituye a `RdYlGn`; los valores fuera de rango van en `#e1e0d9`). `rasterized=True` en los `pcolormesh` grandes.

**Etiquetas:** todas en LaTeX: `$\omega$ (cm$^{-1}$)`, `$T_{xx}$`, `$|\mathrm{CD}|$`, `$R_\mathrm{total} = R_r + R_l$`, `$d_1$, $d_2$, $\theta$, $\varphi$, $f_0$, $\sigma$, $R^2$`.

## 3. Cambios por carpeta

### `Gráficas\` (11 ficheros, editados a mano)
- **1 capa** (`plot_transmitancia`, `Waterplot`, `Waterplot_phi`): bloque de estilo completo; línea negra de referencia; colorbars con `pad` y tamaño coherente; ticks de φ cada 15°.
- **CD** (`plot_cd_tmm`, `theta_freq_line`, `theta_sweep_grid`, `theta_sweep_plot`): colores tab:→paleta de roles (CD azul, R_total rojo, R_l azul/R_r rojo/R_total gris en la cresta); ajuste lineal de la cresta en azul claro `#9ec5f4` sobre inferno; títulos y ejes en LaTeX con acentos; dpi 150→200.
- **Transmision** (`plot_transmision`, `plot_reflexion`, `analisis_anomalias`, `experimento_bicapa_aleatoria`): las 4 componentes con el orden fijo de paleta y cross-pol discontinuo; barras y scatters de anomalías con azul de marca y borde blanco en puntos.
- **Permitividades**: **intacta** (8 scripts) — es estilo B/N de publicación deliberado y ya correcto.

### `Surrogates\CD\` (37 ficheros, parcheados por script con verificación por regla)
- `plot_cd.py` ×9 y `plot_r_total.py` ×9: scatter azul/rojo → **línea TMM negra sólida + línea NN azul discontinua** (mismos datos), TMM dibujado debajo para que el trazo discontinuo revele la coincidencia; figura 9×6→7×4.8; título con `$d_1$, $d_2$, $\theta$` y °.
- `TMM_NN.py` ×18 (CD y R_total): mismo tratamiento dentro de `run_inference`; `figsize=(9,9)`/(9,6)→(7,4.8); guardado a dpi 200.
- `plot_hist_loss.py`: script heredado del doctorado; se restauraron las etiquetas de ejes que estaban comentadas ("Época", "Loss", "$\Delta$ Loss"), curva de validación en azul de marca, leyenda y grid.

### `Optimization\CD\` (19 ficheros)
- `de_optimizer.py` ×9: scatter → líneas con roles (TMM negro / NN azul discontinuo) en los dos paneles; línea de target en gris neutro; ylabels en LaTeX.
- `plot_sweep.py` ×9 + `plot_sweep_elite.py`: corregida la etiqueta física **"λ target (cm⁻¹)" → "$\omega_\mathrm{target}$ (cm$^{-1}$)"**; los 4 paneles de parámetros unificados en azul (un panel = una serie); colorbar y diagonal en LaTeX; heatmap rasterizado.

### `Tandem\T_xx\` (35 ficheros)
- `plot_forward.py` ×5: TMM rojo→negro sólido, NN azul→azul de marca discontinuo, banda ±σ azul al 18 %; ejes en LaTeX.
- `plot_tandem.py` ×5: objetivo negro / reconstrucción TMM azul discontinuo (antes rojo); etiqueta "TMM (parámetros inversa)".
- Demos ×25 (gaussiana, stopband, bandpass, edge; normal y batch): heatmaps R² **RdYlGn → viridis**; en los paneles de ejemplos: objetivo negro, TMM azul sólido, forward NN rojo discontinuo; ventanas de evaluación naranja→gris neutro; banda de stopband/pasabanda en rojo/aqua suaves; unidades cm-1→cm$^{-1}$.
- Los 4 scripts raíz (`plot_*_single`, `plot_training_history`, `plot_arquitectura_forward`) ya eran la referencia: **sin cambios**.

## 4. Verificación

- **Sintaxis:** los 114 scripts de gráficas del repo compilan (`py_compile`) sin errores ni warnings (dos problemas introducidos por el parche —una indentación en un `TMM_NN.py` y raw-strings en los 9 `de_optimizer.py`— se detectaron y corrigieron en esta misma pasada).
- **LaTeX:** las 120 cadenas mathtext únicas de los scripts editados se extrajeron por AST y renderizan sin error.
- **Renderizado real** (ejecutados de verdad y revisados visualmente): `plot_transmision.py`, `plot_transmitancia.py`, `plot_cd_tmm.py`, `plot_sweep.py` (MoO3/MoO3, con los 101 .npz reales), `Surrogates plot_cd.py` (con el modelo .h5 real) y `Tandem plot_forward.py` (grid de 8 paneles con forward.keras). Todos correctos y consistentes entre sí.

## 5. Notas

- Los **PNG antiguos** que haya por el repo siguen siendo los viejos: cada figura se regenera al volver a ejecutar su script.
- Los scripts que solo calculan (`sweep.py`, `evaluar_*.py`, `generate_*`, `train*.py`) no se tocaron.
- Los prints de consola con "cm⁻¹" pueden fallar en consolas cp1252 (comportamiento previo, no introducido ahora); si molesta, ejecutar con `PYTHONIOENCODING=utf-8`.
