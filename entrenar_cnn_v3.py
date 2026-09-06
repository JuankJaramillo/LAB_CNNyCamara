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

from modelo_cnn_v3 import FingerCNNV3
from transforms_cnn_v3 import train_transform_v3, eval_transform_v3


# ============================================================
# CONFIGURACION
# ============================================================

SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 0.0005
MAX_EPOCHS = 30
PATIENCE = 7

TRAIN_PATH = "dataset_landmarks/train"
VAL_PATH = "dataset_landmarks/val"

MODEL_DIR = "models_v3"
RESULTS_DIR = "results_v3"
FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "figures"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_cnn_v3.pth"
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "historial_entrenamiento_v3.csv"
)

LOSS_FIGURE = os.path.join(
    FIGURES_DIR,
    "loss_entrenamiento_v3.png"
)

ACCURACY_FIGURE = os.path.join(
    FIGURES_DIR,
    "accuracy_entrenamiento_v3.png"
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
print("ENTRENAMIENTO CNN V3")
print("==========================================")

print(f"\nDispositivo: {device}")


# ============================================================
# DATASET
# ============================================================
train_dataset = ImageFolder(
    root=TRAIN_PATH,
    transform=train_transform_v3
)

val_dataset = ImageFolder(
    root=VAL_PATH,
    transform=eval_transform_v3
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
    f"Imagenes train: "
    f"{len(train_dataset)}"
)

print(
    f"Imagenes validation: "
    f"{len(val_dataset)}"
)

print(
    f"Clases: "
    f"{train_dataset.classes}"
)


# ============================================================
# MODELO
# ============================================================

modelo = FingerCNNV3(
    num_classes=len(
        train_dataset.classes
    )
)

modelo = modelo.to(device)


# ============================================================
# PARAMETROS DEL MODELO
# ============================================================

total_parametros = sum(
    p.numel()
    for p in modelo.parameters()
)

print(
    f"Parametros del modelo: "
    f"{total_parametros}"
)


# ============================================================
# LOSS Y OPTIMIZADOR
# ============================================================

criterio = nn.CrossEntropyLoss()

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
mejor_val_accuracy = 0.0

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

        predicciones = torch.argmax(
            salidas,
            dim=1
        )

        train_total += (
            etiquetas.size(0)
        )

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

            predicciones = torch.argmax(
                salidas,
                dim=1
            )

            val_total += (
                etiquetas.size(0)
            )

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
    # HISTORIAL
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
    # GUARDAR MEJOR MODELO
    # ========================================================

    if val_loss < mejor_val_loss:

        mejor_val_loss = val_loss
        mejor_val_accuracy = val_accuracy
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
                    128,

                "architecture":
                    "FingerCNNV3",

                "total_parameters":
                    total_parametros
            },
            MODEL_PATH
        )

        modelo_guardado = True

    else:

        epochs_sin_mejora += 1

        modelo_guardado = False


    # ========================================================
    # MOSTRAR RESULTADO
    # ========================================================

    tiempo_epoch = (
        time.time()
        - inicio_epoch
    )

    print(
        f"\nEpoch "
        f"{epoch:02d}/{MAX_EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:   "
        f"{val_loss:.4f} | "
        f"Val Acc:   "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Tiempo: "
        f"{tiempo_epoch:.2f} s"
    )

    if modelo_guardado:

        print(
            ">>> MODELO V3 GUARDADO"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_sin_mejora >= PATIENCE:

        print(
            "\n=========================================="
        )

        print(
            "EARLY STOPPING"
        )

        print(
            "=========================================="
        )

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
    "Perdida durante el entrenamiento - CNN V3"
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
    "Precision durante el entrenamiento - CNN V3"
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

print(
    "\n=========================================="
)

print(
    "ENTRENAMIENTO V3 FINALIZADO"
)

print(
    "=========================================="
)

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
    f"del modelo seleccionado: "
    f"{mejor_val_accuracy:.2f}%"
)

print(
    f"Tiempo total: "
    f"{tiempo_total:.2f} segundos"
)

print(
    f"Parametros: "
    f"{total_parametros}"
)

print(
    "\nModelo guardado en:"
)

print(
    MODEL_PATH
)

print(
    "\nHistorial guardado en:"
)

print(
    CSV_PATH
)

print(
    "\nEl conjunto TEST NO fue utilizado."
)