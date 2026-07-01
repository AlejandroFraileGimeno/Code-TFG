# Transmisión T_ss — Red neuronal

Red neuronal (TensorFlow/Keras 3) que predice el espectro de **transmitancia
`T_ss = |t_ss|²`** (componente s→s, polarización lineal) de una **bicapa de MoO₃
con ambas capas rotadas**, sobre substrato de **BaF₂**, a **incidencia normal
(α = 0)**.

Mismo esquema que el proyecto de dicroísmo circular (carpeta `ParaAlejandro`),
pero con su propia física y datos.

## Física

| Elemento | Valor |
|---|---|
| Magnitud | `T_ss = calculate_transmission(ω, α, estructura, basis="linear")[1]` |
| Incidencia | α = 0° (normal) |
| Superestrato | Aire |
| Substrato | BaF₂ |
| Capas | MoO₃(d1, φ=θ1) + MoO₃(d2, φ=θ2) |

## La red

- **Entradas (5):** `θ1, θ2, d1, d2, frecuencia`
- **Salida (1):** `T_ss` en esa frecuencia
- Rangos: θ1, θ2 ∈ [0, 180]° · d1, d2 ∈ [200, 2000] nm · ν ∈ [600, 1100] cm⁻¹

## Flujo (lanzadores de VS Code con prefijo `[T_ss]`)

| Paso | Script | Qué hace |
|------|--------|----------|
| 1 | `generate_dataset_parallel.py` | Genera el dataset en `NN_Code/Dataset_Tss_Bilayer` (10000 estructuras, paralelo) |
| 2 | `train.py` | Entrena y guarda el modelo en `NN_Code/Tss_Models_Trained_bilayers_MoO3/Model_1seed/` |
| 3 | `TMM_NN.py` | Compara T_ss de la red vs TMM y guarda PNGs en `.../results/` |
| 4 | `evaluar_modelo.py` | Imprime MAE, RMSE y correlación frente al cálculo físico |
| 5 | `plot_history.py` | Dibuja la curva de pérdida (train vs val) |
| 6 | `plot_tss.py` | Pinta el T_ss del modelo; te pide θ1, θ2, d1, d2 por teclado |

## Archivos

```
Transmision_Tss/
├── generalized_transfer_matrix_method/   # Paquete GTMM (autocontenido)
├── src/data_generation.py                # Motor de generación (T_ss, 2 ángulos)
├── forward_model.py                       # Modelo Keras (5 entradas)
├── training_forward.py                    # Lógica de entrenamiento (nfeatures=5)
├── utils_nn_forward.py                    # Datos / normalización / predicción
├── generate_dataset_parallel.py           # [1] generar dataset (rápido)
├── generate_database.py                   #     generar dataset (serie, lento)
├── train.py                               # [2] entrenar
├── TMM_NN.py                              # [3] inferencia NN vs TMM
├── evaluar_modelo.py                      # [4] métricas
├── plot_history.py                        # [5] curva de entrenamiento
├── plot_tss.py                            # [6] pintar T_ss del modelo
├── SEED_LIST.csv
└── requirements.txt
```

> Comparte el mismo `.venv` que el proyecto de CD (está en la raíz `Código Lucía`).
