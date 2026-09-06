import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

from modelo_cnn_v3 import FingerCNNV3
from transforms_cnn_v3 import eval_transform_v3


# ============================================================
# CONFIGURACION
# ============================================================

TEST_PATH = "dataset_landmarks/test"

MODEL_PATH = "models_v3/best_cnn_v3.pth"

RESULTS_PATH = "results_final"

BATCH_SIZE = 32

CLASES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]


# ============================================================
# CREAR CARPETA DE RESULTADOS
# ============================================================

os.makedirs(
    RESULTS_PATH,
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
print("EVALUACION FINAL - TEST INDEPENDIENTE")
print("==========================================")

print(
    f"\nDispositivo: {device}"
)


# ============================================================
# CARGAR TEST
# ============================================================

test_dataset = ImageFolder(
    TEST_PATH,
    transform=eval_transform_v3
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print()
print(
    f"Clases detectadas: {test_dataset.classes}"
)

print(
    f"Total imagenes test: {len(test_dataset)}"
)


# ============================================================
# CARGAR MODELO
# ============================================================

modelo = FingerCNNV3(
    num_classes=5
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


modelo.load_state_dict(
    checkpoint["model_state_dict"]
)


modelo = modelo.to(
    device
)


modelo.eval()


print()
print("MODELO SELECCIONADO:")
print("------------------------------------------")

print(
    f"Epoch: {checkpoint['epoch']}"
)

print(
    f"Validation Loss: "
    f"{checkpoint['val_loss']:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)


# ============================================================
# INFERENCIA SOBRE TEST
# ============================================================

y_true = []
y_pred = []

confianzas = []
rutas_imagenes = []


print()
print(
    "Evaluando test..."
)


indice_global = 0


with torch.no_grad():

    for imagenes, etiquetas in test_loader:

        imagenes = imagenes.to(
            device
        )

        etiquetas = etiquetas.to(
            device
        )


        logits = modelo(
            imagenes
        )


        probabilidades = torch.softmax(
            logits,
            dim=1
        )


        confianza_batch, predicciones = torch.max(
            probabilidades,
            dim=1
        )


        y_true.extend(
            etiquetas.cpu().numpy()
        )

        y_pred.extend(
            predicciones.cpu().numpy()
        )

        confianzas.extend(
            confianza_batch.cpu().numpy()
        )


        for _ in range(
            len(etiquetas)
        ):

            ruta, _ = test_dataset.samples[
                indice_global
            ]

            rutas_imagenes.append(
                ruta
            )

            indice_global += 1


y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)

confianzas = np.array(
    confianzas
)


# ============================================================
# ACCURACY GLOBAL
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


print()
print("==========================================")
print("RESULTADOS TEST")
print("==========================================")

print(
    f"\nAccuracy global: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# METRICAS POR CLASE
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
            3,
            4
        ],
        zero_division=0
    )
)


print()
print("METRICAS POR CLASE:")
print("------------------------------------------")

print(
    f"{'Clase':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Support':<10}"
)


for i in range(
    5
):

    print(
        f"{CLASES[i]:<8}"
        f"{precision[i]:<12.4f}"
        f"{recall[i]:<12.4f}"
        f"{f1[i]:<12.4f}"
        f"{support[i]:<10}"
    )


# ============================================================
# MATRIZ DE CONFUSION
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1,
        2,
        3,
        4
    ]
)


print()
print("MATRIZ DE CONFUSION:")
print("------------------------------------------")

print(
    cm
)


# ============================================================
# GUARDAR MATRIZ DE CONFUSION - CONTEOS
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        7
    )
)


imagen = ax.imshow(
    cm
)


ax.set_title(
    "Matriz de confusión - Test independiente"
)


ax.set_xlabel(
    "Clase predicha"
)


ax.set_ylabel(
    "Clase real"
)


ax.set_xticks(
    np.arange(
        len(CLASES)
    )
)


ax.set_yticks(
    np.arange(
        len(CLASES)
    )
)


ax.set_xticklabels(
    CLASES
)


ax.set_yticklabels(
    CLASES
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        ax.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center"
        )


fig.colorbar(
    imagen,
    ax=ax
)


plt.tight_layout()


ruta_matriz = os.path.join(
    RESULTS_PATH,
    "matriz_confusion_test.png"
)


plt.savefig(
    ruta_matriz,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# MATRIZ DE CONFUSION NORMALIZADA
# ============================================================

cm_normalizada = (
    cm.astype(float)
    / cm.sum(
        axis=1,
        keepdims=True
    )
)


fig, ax = plt.subplots(
    figsize=(
        8,
        7
    )
)


imagen = ax.imshow(
    cm_normalizada
)


ax.set_title(
    "Matriz de confusión normalizada - Test"
)


ax.set_xlabel(
    "Clase predicha"
)


ax.set_ylabel(
    "Clase real"
)


ax.set_xticks(
    np.arange(
        len(CLASES)
    )
)


ax.set_yticks(
    np.arange(
        len(CLASES)
    )
)


ax.set_xticklabels(
    CLASES
)


ax.set_yticklabels(
    CLASES
)


for i in range(
    cm_normalizada.shape[0]
):

    for j in range(
        cm_normalizada.shape[1]
    ):

        ax.text(
            j,
            i,
            f"{cm_normalizada[i, j] * 100:.1f}%",
            ha="center",
            va="center"
        )


fig.colorbar(
    imagen,
    ax=ax
)


plt.tight_layout()


ruta_matriz_norm = os.path.join(
    RESULTS_PATH,
    "matriz_confusion_test_normalizada.png"
)


plt.savefig(
    ruta_matriz_norm,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# GUARDAR METRICAS CSV
# ============================================================

ruta_metricas = os.path.join(
    RESULTS_PATH,
    "metricas_test.csv"
)


with open(
    ruta_metricas,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    writer = csv.writer(
        archivo
    )


    writer.writerow(
        [
            "clase",
            "precision",
            "recall",
            "f1_score",
            "support"
        ]
    )


    for i in range(
        5
    ):

        writer.writerow(
            [
                CLASES[i],
                f"{precision[i]:.6f}",
                f"{recall[i]:.6f}",
                f"{f1[i]:.6f}",
                int(
                    support[i]
                )
            ]
        )


# ============================================================
# GUARDAR PREDICCIONES INDIVIDUALES
# ============================================================

ruta_predicciones = os.path.join(
    RESULTS_PATH,
    "predicciones_test.csv"
)


with open(
    ruta_predicciones,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    writer = csv.writer(
        archivo
    )


    writer.writerow(
        [
            "imagen",
            "clase_real",
            "clase_predicha",
            "confianza",
            "correcta"
        ]
    )


    for ruta, real, pred, conf in zip(
        rutas_imagenes,
        y_true,
        y_pred,
        confianzas
    ):

        writer.writerow(
            [
                ruta,
                CLASES[
                    real
                ],
                CLASES[
                    pred
                ],
                f"{conf * 100:.2f}",
                int(
                    real == pred
                )
            ]
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

reporte = classification_report(
    y_true,
    y_pred,
    target_names=CLASES,
    digits=4,
    zero_division=0
)


print()
print("CLASSIFICATION REPORT:")
print("------------------------------------------")

print(
    reporte
)


# ============================================================
# GUARDAR RESUMEN TXT
# ============================================================

ruta_resumen = os.path.join(
    RESULTS_PATH,
    "resumen_test.txt"
)


with open(
    ruta_resumen,
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(
        "EVALUACION FINAL - TEST INDEPENDIENTE\n"
    )

    archivo.write(
        "=====================================\n\n"
    )


    archivo.write(
        f"Checkpoint seleccionado: epoch "
        f"{checkpoint['epoch']}\n"
    )


    archivo.write(
        f"Validation Loss: "
        f"{checkpoint['val_loss']:.4f}\n"
    )


    archivo.write(
        f"Validation Accuracy: "
        f"{checkpoint['val_accuracy']:.2f}%\n\n"
    )


    archivo.write(
        f"Total muestras test: "
        f"{len(y_true)}\n"
    )


    archivo.write(
        f"Accuracy test: "
        f"{accuracy * 100:.2f}%\n\n"
    )


    archivo.write(
        "Matriz de confusion:\n"
    )

    archivo.write(
        str(
            cm
        )
    )

    archivo.write(
        "\n\n"
    )


    archivo.write(
        "Classification report:\n"
    )

    archivo.write(
        reporte
    )


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print("==========================================")
print("ARCHIVOS GENERADOS")
print("==========================================")

print(
    ruta_matriz
)

print(
    ruta_matriz_norm
)

print(
    ruta_metricas
)

print(
    ruta_predicciones
)

print(
    ruta_resumen
)


print()
print("==========================================")
print("EVALUACION FINAL COMPLETADA")
print("==========================================")