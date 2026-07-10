# Code TFG — Optimización de bicapas ópticas mediante redes neuronales

Código del Trabajo de Fin de Grado. El objetivo es encontrar la geometría óptima (ángulo y espesores) de bicapas de materiales anisotrópicos que maximice el dicroísmo circular (CD) usando redes neuronales como modelos sustitutos (*surrogates*) del cálculo físico exacto (TMM).

## Materiales estudiados

Se trabaja con bicapas formadas por combinaciones de tres materiales birrefringentes:

- **MoO₃** — trióxido de molibdeno
- **V₂O₅** — pentóxido de vanadio
- **MgTeMoO₆** — teluro de molibdato de magnesio

Hay **9 combinaciones** de bicapas (capa1 / capa2), con estructura: `Air / capa1(d₁, θ) / capa2(d₂) / Au`.

## Estructura del repositorio

```
Code TFG/
│
├── TMM/                          # Paquete de cálculo físico (Transfer Matrix Method)
│   └── generalized_transfer_matrix_method/
│       ├── tmm.py                # Cálculo de transmisión/reflexión
│       ├── permittivities.py     # Permitividades de cada material
│       └── Permittivities/       # Datos ópticos tabulados
│
├── Datasets/
│   └── CD/
│       └── MoO3_V2O5/            # (ejemplo — hay 9 combinaciones)
│           ├── CD_spectra_norm.csv   # Espectros de CD normalizado
│           ├── R_total_spectra.csv   # Espectros de reflectancia total
│           ├── angles.csv            # Ángulos de rotación (θ)
│           └── thickness.csv         # Espesores (d₁, d₂) en nm
│
├── Surrogates/
│   └── CD/
│       └── MoO3, V2O5/           # (ejemplo — hay 9 combinaciones)
│           ├── Generación/       # Scripts para generar el dataset con TMM
│           ├── Train/            # Entrenamiento de las redes neuronales
│           │   ├── train.py          # Entrena modelo de CD_norm
│           │   └── train_r_total.py  # Entrena modelo de R_total
│           ├── Evaluación/       # Comparación NN vs TMM para CD_norm
│           └── Evaluación R_total/   # Comparación NN vs TMM para R_total
│
├── Models/
│   ├── CD/
│   │   └── MoO3_V2O5/            # (ejemplo — hay 9 combinaciones)
│   │       ├── Model_1seed/      # Modelo entrenado (seed 1 de 5)
│   │       │   ├── Model_1seed.h5    # Pesos de la red neuronal (Keras 3)
│   │       │   ├── scalers.json      # Normalizadores de entrada/salida
│   │       │   └── hyperparameters.txt
│   │       └── ...               # Model_2seed ... Model_5seed
│   └── R_total/
│       └── MoO3_V2O5/            # Ídem para el modelo de R_total
│
├── Optimization/
│   └── CD/
│       └── MoO3_V2O5/            # (ejemplo — hay 9 combinaciones)
│           └── DE/
│               ├── de_optimizer.py   # Optimización por Evolución Diferencial
│               └── results/          # PNGs con el espectro óptimo encontrado
│
└── Seed/
    └── SEED_LIST.csv             # Semillas aleatorias para reproducibilidad
```

> **Nota:** `Datasets/` no se versiona (son ~4 GB de espectros). Se regeneran con `generate_database.py` (paso 1). Los modelos ya entrenados **sí** se incluyen en `Models/`.

## Flujo de trabajo

Para cada combinación de materiales, el proceso sigue estos pasos:

```
1. Generar dataset  →  2. Entrenar NN  →  3. Evaluar NN  →  4. Optimizar con DE
```

| Paso | Carpeta | Script | Qué hace |
|------|---------|--------|----------|
| 1 | `Surrogates/CD/<par>/Generación/` | `generate_database.py` | Calcula 10 000 estructuras con TMM y guarda los espectros en `Datasets/` |
| 2a | `Surrogates/CD/<par>/Train/` | `train.py` | Entrena red neuronal para predecir CD_norm |
| 2b | `Surrogates/CD/<par>/Train/` | `train_r_total.py` | Entrena red neuronal para predecir R_total |
| 3a | `Surrogates/CD/<par>/Evaluación/` | `TMM_NN.py` / `evaluar_modelo.py` | Compara predicciones NN vs TMM (CD_norm) |
| 3b | `Surrogates/CD/<par>/Evaluación R_total/` | `TMM_NN.py` / `evaluar_modelo.py` | Compara predicciones NN vs TMM (R_total) |
| 4 | `Optimization/CD/<par>/DE/` | `de_optimizer.py` | Busca (θ, d₁, d₂) óptimos usando Evolución Diferencial con las NNs |

## Función de mérito (FoM)

El optimizador maximiza:

```
FoM = C1 × CD_norm_peak + C2 × R_total_at_peak
```

donde `C1 >> C2` (por defecto `C1=1.0`, `C2=0.1`), priorizando el pico de CD sobre la reflectancia total.

## Configuración del entorno

**Requisitos:** Python 3.12 + TensorFlow 2.x (Keras 3)

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install tensorflow numpy scipy matplotlib
```

> En Windows, TensorFlow solo ejecuta en **CPU**. El entrenamiento completo puede tardar; para pruebas rápidas reduce `n_data` en `generate_database.py` o `NUM_SEEDS` en `de_optimizer.py`.

## Modelos entrenados

Cada combinación tiene **5 modelos** (seeds 1–5) para estimar la variabilidad. Durante la optimización se usa un ensemble promediando las predicciones de los N seeds configurados (`NUM_SEEDS` en `de_optimizer.py`).

Los modelos ya están entrenados y guardados en `Models/`. Solo hay que volver al paso 1–2 si se quieren regenerar.
