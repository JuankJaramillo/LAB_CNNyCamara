import os
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from modelo_cnn import FingerCNN
from transforms_cnn import train_transform, eval_transform


# ============================================================
# CONFIGURACION
# ============================================================

SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 5

TRAIN_PATH = "dataset_v2/train"
VAL_PATH = "dataset_v2/val"

MODEL_DIR = "models_v2"
RESULTS_DIR = "results_v2"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_cnn_v2.pth"
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "historial_entrenamiento.csv"
)

LOSS_FIGURE = os.path.join(
    FIGURES_DIR,
    "loss_entrenamiento.png"
)

ACCURACY_FIGURE = os.path.join(
    FIGURES_DIR,
    "accuracy_entrenamiento.png"
)


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CREAR CARPETAS
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("==========================================")
print("ENTRENAMIENTO CNN")
print("==========================================")

print(f"\nDispositivo: {device}")


# ============================================================
# DATASET
# ============================================================

train_dataset = ImageFolder(
    root=TRAIN_PATH,
    transform=train_transform
)

val_dataset = ImageFolder(
    root=VAL_PATH,
    transform=eval_transform
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    f"Imagenes train: {len(train_dataset)}"
)

print(
    f"Imagenes validation: {len(val_dataset)}"
)

print(
    f"Clases: {train_dataset.classes}"
)


# ============================================================
# MODELO
# ============================================================

modelo = FingerCNN(
    num_classes=len(train_dataset.classes)
)

modelo = modelo.to(device)


# ============================================================
# FUNCION DE PERDIDA
# ============================================================

criterio = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZADOR
# ============================================================

optimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# HISTORIAL
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ============================================================
# EARLY STOPPING
# ============================================================

mejor_val_loss = float("inf")
mejor_epoch = 0
epochs_sin_mejora = 0


# ============================================================
# TIEMPO TOTAL
# ============================================================

inicio_total = time.time()


# ============================================================
# ENTRENAMIENTO
# ============================================================

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    inicio_epoch = time.time()

    # ========================================================
    # TRAIN
    # ========================================================

    modelo.train()

    train_loss_acumulada = 0.0
    train_correctas = 0
    train_total = 0

    for imagenes, etiquetas in train_loader:

        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        optimizador.zero_grad()

        salidas = modelo(
            imagenes
        )

        loss = criterio(
            salidas,
            etiquetas
        )

        loss.backward()

        optimizador.step()

        train_loss_acumulada += (
            loss.item()
            * imagenes.size(0)
        )

        _, predicciones = torch.max(
            salidas,
            1
        )

        train_total += etiquetas.size(0)

        train_correctas += (
            predicciones == etiquetas
        ).sum().item()


    train_loss = (
        train_loss_acumulada
        / train_total
    )

    train_accuracy = (
        100.0
        * train_correctas
        / train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    modelo.eval()

    val_loss_acumulada = 0.0
    val_correctas = 0
    val_total = 0

    with torch.no_grad():

        for imagenes, etiquetas in val_loader:

            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            salidas = modelo(
                imagenes
            )

            loss = criterio(
                salidas,
                etiquetas
            )

            val_loss_acumulada += (
                loss.item()
                * imagenes.size(0)
            )

            _, predicciones = torch.max(
                salidas,
                1
            )

            val_total += etiquetas.size(0)

            val_correctas += (
                predicciones == etiquetas
            ).sum().item()


    val_loss = (
        val_loss_acumulada
        / val_total
    )

    val_accuracy = (
        100.0
        * val_correctas
        / val_total
    )


    # ========================================================
    # GUARDAR HISTORIAL
    # ========================================================

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # ========================================================
    # MEJOR MODELO
    # ========================================================

    mejora = (
        val_loss < mejor_val_loss
    )

    if mejora:

        mejor_val_loss = val_loss
        mejor_epoch = epoch
        epochs_sin_mejora = 0

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    modelo.state_dict(),

                "optimizer_state_dict":
                    optimizador.state_dict(),

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "classes":
                    train_dataset.classes,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "img_size":
                    128
            },
            MODEL_PATH
        )

        estado_modelo = "MODELO GUARDADO"

    else:

        epochs_sin_mejora += 1
        estado_modelo = ""


    # ========================================================
    # RESULTADO DEL EPOCH
    # ========================================================

    tiempo_epoch = (
        time.time()
        - inicio_epoch
    )

    print(
        f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:   {val_accuracy:.2f}%"
    )

    print(
        f"Tiempo: {tiempo_epoch:.2f} s"
    )

    if estado_modelo != "":

        print(
            f">>> {estado_modelo}"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_sin_mejora >= PATIENCE:

        print("\n==========================================")
        print("EARLY STOPPING")
        print("==========================================")

        print(
            f"No hubo mejora durante "
            f"{PATIENCE} epochs."
        )

        break


# ============================================================
# TIEMPO TOTAL
# ============================================================

tiempo_total = (
    time.time()
    - inicio_total
)


# ============================================================
# GUARDAR CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as archivo_csv:

    writer = csv.writer(
        archivo_csv
    )

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy"
    ])

    for i in range(
        len(train_losses)
    ):

        writer.writerow([
            i + 1,
            train_losses[i],
            val_losses[i],
            train_accuracies[i],
            val_accuracies[i]
        ])


# ============================================================
# GRAFICA LOSS
# ============================================================

epochs = range(
    1,
    len(train_losses) + 1
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Perdida durante el entrenamiento"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    LOSS_FIGURE,
    dpi=200
)

plt.show()


# ============================================================
# GRAFICA ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Precision durante el entrenamiento"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    ACCURACY_FIGURE,
    dpi=200
)

plt.show()


# ============================================================
# RESUMEN
# ============================================================

print("\n==========================================")
print("ENTRENAMIENTO FINALIZADO")
print("==========================================")

print(
    f"Epochs ejecutadas: "
    f"{len(train_losses)}"
)

print(
    f"Mejor epoch: "
    f"{mejor_epoch}"
)

print(
    f"Mejor Validation Loss: "
    f"{mejor_val_loss:.4f}"
)

print(
    f"Validation Accuracy "
    f"del mejor modelo: "
    f"{val_accuracies[mejor_epoch - 1]:.2f}%"
)

print(
    f"Tiempo total: "
    f"{tiempo_total:.2f} segundos"
)

print(
    f"\nModelo guardado en:"
)

print(
    MODEL_PATH
)

print(
    f"\nHistorial guardado en:"
)

print(
    CSV_PATH
)

print(
    "\nEl conjunto TEST no fue utilizado."
)