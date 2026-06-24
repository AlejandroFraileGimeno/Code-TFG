# Código Lucía — Red neuronal de dicroísmo circular (CD)

Red neuronal (TensorFlow/Keras 3) que predice espectros de dicroísmo circular en
bicapas de MoO₃, apoyada en el paquete `generalized_transfer_matrix_method` (GTMM).

## Estructura

```
Código Lucía/
├── .venv/                      # Entorno virtual (NO tocar, no se sincroniza)
├── .vscode/                    # Configuración de VS Code (intérprete, lanzadores)
├── .env                        # Variables de entorno (silencian logs de TensorFlow)
└── ParaAlejandro/
    ├── generalized_transfer_matrix_method/   # Paquete GTMM (autocontenido)
    ├── src/data_generation.py  # Generador de datos (reconstruido)
    ├── forward_model.py        # Definición del modelo Keras
    ├── training_forward.py     # Lógica de entrenamiento
    ├── utils_nn_forward.py     # Utilidades (datos, normalización, predicción)
    ├── SEED_LIST.csv           # Semillas aleatorias
    ├── requirements.txt        # Dependencias
    │
    ├── generate_dataset_parallel.py  # [1] Generar dataset (rápido, ~30 min)
    ├── generate_database.py          #     Generar dataset (serie, original, ~3 h)
    ├── train.py                      # [2] Entrenar el modelo
    ├── TMM_NN.py                      # [3] Inferencia: comparar NN vs TMM (PNGs)
    ├── evaluar_modelo.py             # [4] Métricas de calidad (MAE/RMSE/corr)
    └── plot_history.py               # [5] Graficar la curva de entrenamiento
```

## Cómo ejecutar (en VS Code)

1. Abre la carpeta **Código Lucía** en VS Code.
2. Comprueba que abajo a la derecha aparece el intérprete `.venv`. Si no:
   `Ctrl+Shift+P` → *Python: Select Interpreter* → *Enter interpreter path* →
   pega: `.venv\Scripts\python.exe` (o selecciónalo de la lista).
3. Pulsa **F5** y elige una de las configuraciones (1 a 5). Cada una ya corre
   con el `.venv` y desde la carpeta correcta.

## Flujo de trabajo

| Paso | Script | Qué hace |
|------|--------|----------|
| 1 | `generate_dataset_parallel.py` | Genera el dataset en `NN_Code/Dataset_MoO3_Bilayer` (10000 estructuras, paralelo) |
| 2 | `train.py` | Entrena y guarda el modelo en `NN_Code/Forward_Models_Trained_bilayers_MoO3/Model_1seed/` |
| 3 | `TMM_NN.py` | Compara espectros NN vs TMM y guarda PNGs en `.../results/` |
| 4 | `evaluar_modelo.py` | Imprime MAE, RMSE y correlación frente al cálculo físico |
| 5 | `plot_history.py` | Dibuja la curva de pérdida (train vs val) |

> El dataset ya está generado y el modelo ya está entrenado. Solo necesitas
> volver al paso 1/2 si quieres rehacerlos.

## Notas técnicas

- **Python 3.12 + TensorFlow 2.21 (Keras 3).** El modelo se entrena, guarda
  (`.h5`) y carga con Keras 3. No se usa `tf-keras` ni `TF_USE_LEGACY_KERAS`.
- **GPU:** TensorFlow en Windows nativo funciona solo en **CPU**. El
  entrenamiento completo (8M muestras) tarda; para pruebas rápidas baja
  `ntrain`/`nvalidation` en `train.py`.
- **Rutas con tildes:** la carpeta se llama "Código"; Keras 3 maneja bien la
  tilde al cargar el `.h5` (Keras 2 no, por eso no se usa).
- El paquete GTMM y el módulo `src/data_generation.py` no venían en el zip
  original: el primero se copió desde otra carpeta y el segundo se reconstruyó
  a partir del resto del código.
