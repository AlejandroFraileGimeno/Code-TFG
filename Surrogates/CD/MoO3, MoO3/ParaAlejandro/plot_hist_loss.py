# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:35:38 2024

@author: nanooptics
"""

import matplotlib.pyplot as plt
import numpy as np
import ast

seed = 0
# list_simulations = [i+2 for i in range(seed)]
max_Nbranches = 10
min_Nbranches = 1
list_Nbranches = [i for i in range(max_Nbranches, min_Nbranches - 1, -1)]
print(list_Nbranches)
# directory = r'C:\Users\nanooptics\OneDrive - Universidad de Oviedo\DOCTORADO\Redes Zaragoza\Actuales\codes_nn_ramas\models_fantasy_minuit_ReIm'
# directory = r'C:\Users\nanooptics\OneDrive - Universidad de Oviedo\DOCTORADO\Redes Zaragoza\Actuales\codes_nn_ramas\prueba_models_fantasy_minuit_Re'
# directory = r'C:\Users\nanooptics\OneDrive - Universidad de Oviedo\DOCTORADO\PROGRAMS\Redes Lucia\Modelos\models_inverse_minuit_Re_bicapas'
# directory = r'C:\Users\nanooptics\OneDrive - Universidad de Oviedo\DOCTORADO\PROGRAMS\Redes Lucia\Modelos\models_inverse_minuit_Re_bicapas'
# directory = r'C:\Users\nanooptics\OneDrive - Universidad de Oviedo\DOCTORADO\PROGRAMS\Redes Lucia\Modelos\Redes 917\models_inverse_design_minuit_Re'
# directory = r'C:\Users\nanooptics\OneDrive - Universidad de Oviedo\DOCTORADO\PROGRAMS\Redes Lucia\Modelos\models_fantasy_minuit_Re/Con mas capas'


min_train = []
min_val = []
individual = False
cabezas = True
IFC = False

if IFC:
    directory = r".\Modelos\Redes 917\models_IFC_Lucia_Luis\models_IFC_filtrado"

    modelo = "Model_" + str(seed) + "seed_filtrado_ps8"
    # carpeta_guardar = '/RESULTADOS ENTRENAMIENTO'
    # Lee el archivo de texto
    with open(directory + "/" + modelo + "/history_loss.txt", "r") as file:
        lines = file.readlines()
    # Separa las columnas
    epochs = [float(line.split()[0]) for line in lines]
    evaluation_loss = [float(line.split()[1]) for line in lines]
    training_loss = [float(line.split()[2]) for line in lines]
    min_train.append(np.min(evaluation_loss))
    print("Min loss in training curve for " + modelo)
    print(np.min(training_loss))
    print("Min loss in validation curve for " + modelo)
    print(np.min(evaluation_loss))
    print("\n")
    # Crea el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, evaluation_loss, label="Validation curve")
    plt.plot(epochs, training_loss, label="Training curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.title("Historial de Pérdida " + modelo)
    plt.legend()
    plt.grid(True)

    filename = directory + "./results_" + modelo + ".png"
    plt.savefig(filename)

    plt.show()
# Para un entrenamiento concreto

if individual:
    directory = r"C:\Users\nanooptics\OneDrive - Universidad de Oviedo\DOCTORADO\PROGRAMS\Redes Lucia\Modelos\Redes 660\seguimiento\models_inverse_desing_minuit_freq"
    for j in list_Nbranches:
        modelo = "Model_" + str(seed) + "seed_" + str(j) + "branches"
        carpeta_guardar = "/RESULTADOS ENTRENAMIENTO"
        # Lee el archivo de texto
        with open(directory + "/" + modelo + "/history_loss.txt", "r") as file:
            lines = file.readlines()
        # Separa las columnas
        epochs = [float(line.split()[0]) for line in lines]
        evaluation_loss = [float(line.split()[1]) for line in lines]
        training_loss = [float(line.split()[2]) for line in lines]
        min_train.append(np.min(evaluation_loss))
        print("Min loss in training curve for " + modelo)
        print(np.min(training_loss))
        print("Min loss in validation curve for " + modelo)
        print(np.min(evaluation_loss))
        print("\n")
        # Crea el gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, evaluation_loss, label="Validation curve")
        plt.plot(epochs, training_loss, label="Training curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.title("Historial de Pérdida " + modelo)
        plt.legend()
        plt.grid(True)

        filename = (
            directory
            + "/Model_"
            + str(seed)
            + "seed_"
            + str(j)
            + "branches"
            + "./results_"
            + modelo
            + ".png"
        )
        # plt.savefig(filename)

        plt.show()

# Varios entrenamientos
if cabezas:
    directory = r"C:\Users\Lucia\Doctorado\Proyectos Git\Proyecto Thermal Emission Papadakis\GeneralizedTransferMatrixMethodPy\NN_Code\Models_Trained_bilayers_MoO3"
    epo = []
    ev_loss = []
    train_loss = []
    ultima = 0
    for j in list_Nbranches:
        modelo = "Model_" + str(seed) + "seed_" + str(j) + "branches"
        # Lee el archivo de texto
        with open(directory + "/" + modelo + "/history_loss.txt", "r") as file:
            lines = file.readlines()
        # Separa las columnas
        epochs = [float(line.split()[0]) + ultima + 1 for line in lines]
        evaluation_loss = [float(line.split()[1]) for line in lines]
        training_loss = [float(line.split()[2]) for line in lines]
        min_val.append(np.min(evaluation_loss))
        min_train.append(np.min(training_loss))
        epo.append(epochs)
        ev_loss.append(evaluation_loss)
        train_loss.append(training_loss)
        ultima = epochs[-1]

        if j < 1:
            f = open(directory + "/" + modelo + "/activity.txt", "r")
            a = f.readline()
            a_list = ast.literal_eval(a)
            a_array = np.array(a_list)
            print(a_array)
            print(j)
            plt.figure(figsize=(10, 6))
            plt.xticks(np.arange(0, j, 1))
            plt.bar(
                np.arange(0, j, 1),
                a_array,
                width=1.0,
                align="center",
                edgecolor="black",
            )
            # Personalizar el gráfico
            plt.title("Distribución del número de ramas utilizadas")
            plt.xlabel("Cabezas")
            plt.ylabel("Nº de aciertos")
            plt.grid(True)
            plt.savefig(
                directory + "/" + modelo + "/Histograma" + str(j) + " Cabezas.png",
                bbox_inches="tight",
                dpi=500,
            )
            plt.show()

    epo = np.concatenate(epo)
    ev_loss = np.concatenate(ev_loss)
    train_loss = np.concatenate(train_loss)

    # Crea el gráfico en funcion de épocas
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(epo, ev_loss, c="k", label="Validation curve")
    # plt.plot(epo, train_loss, label='Training curve')
    # plt.xlabel('Epoch',fontsize=20)
    # plt.ylabel('Loss value',fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    # plt.title('Historial de Pérdida '+modelo)
    # plt.legend(fontsize=16)
    plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    filename = directory + "./PaperTresults_epocas" + modelo + ".png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    # # Crea el gráfico en función de las cabezas con delta de loss
    # plt.figure(figsize=(10, 6))
    # delta_train = np.array(train_loss)[1:]-np.array(train_loss)[:-1]
    # delta_val = np.array(ev_loss)[1:] - np.array(ev_loss)[:-1]
    # plt.plot(epo[:-1], delta_train, label='Training curve')
    # plt.plot(epo[:-1], delta_val, label='Validation curve')
    # plt.xlabel('Epoch')
    # plt.ylabel(r' $ \Delta $ Loss value')
    # plt.title('Historial de Pérdida '+modelo)
    # plt.legend()
    # filename = directory+ './results_delta_epocas_final_'+modelo+'.png'
    # plt.savefig(filename)
    # plt.show()

    # # Crea el gráfico en función de las cabezas
    # plt.figure(figsize=(8, 4))
    # plt.xticks(np.arange(0, max_Nbranches, 1))
    # plt.plot(list_Nbranches, min_train, label='Training curve')
    # plt.plot(list_Nbranches, min_val, label='Validation curve')
    # plt.xlim([max(list_Nbranches), min(list_Nbranches)])
    # plt.xlabel('M',fontsize=17)
    # plt.ylabel('Loss value',fontsize=17)
    # plt.tick_params(axis='both', which='major', labelsize=17)
    # # plt.title('Historial de Pérdida de cabezas '+modelo)
    # # plt.legend()
    # # plt.grid(True)
    # # plt.xticks(list_Nbranches_reversed)
    # filename = directory+ './Tresults_cabezas_final_'+modelo+'.png'
    # plt.savefig(filename, bbox_inches='tight')
    # plt.show()

    # Crea el gráfico en función de las cabezas con delta de loss
    plt.figure(figsize=(8, 6), dpi=100)
    plt.xticks(np.arange(0, max_Nbranches - 1, 1))
    delta_train = np.array(min_train)[1:] - np.array(min_train)[:-1]
    delta_val = np.array(min_val)[1:] - np.array(min_val)[:-1]
    # plt.plot(list_Nbranches[:-1], delta_train, linestyle='-.', marker='o', color = 'k', label='Training curve')
    plt.plot(
        list_Nbranches[:-1],
        delta_val,
        linestyle="--",
        marker=".",
        color="k",
        label="Validation curve",
    )
    plt.xlim([max(list_Nbranches[:-1]), min(list_Nbranches[:-1])])
    # plt.xlabel('M',fontsize=15)
    # plt.ylabel(r' $ \Delta $ Loss value',fontsize=15)
    plt.xticks(np.arange(0, 21, 2))
    plt.tick_params(axis="both", which="major", labelsize=15)
    # plt.title('Historial de Pérdida de cabezas '+modelo)
    # plt.legend(fontsize=16)
    # plt.grid(True)
    # plt.xticks(list_Nbranches_reversed)
    filename = directory + "./PaperTresults_delta_cabezas_final_" + modelo + ".png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
