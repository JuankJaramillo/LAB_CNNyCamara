import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from modelo_cnn import FingerCNN
from transforms_cnn import eval_transform


# ============================================================
# CONFIGURACION
# ============================================================

BATCH_SIZE = 32

MODEL_PATH = "models_v2/best_cnn_v2.pth"

TRAIN_PATH = "dataset_v2/train"
VAL_PATH = "dataset_v2/val"


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = ImageFolder(
    root=TRAIN_PATH,
    transform=eval_transform
)

val_dataset = ImageFolder(
    root=VAL_PATH,
    transform=eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CARGAR CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

modelo = FingerCNN(
    num_classes=5
)

modelo.load_state_dict(
    checkpoint["model_state_dict"]
)

modelo = modelo.to(device)

modelo.eval()


print("==========================================")
print("DIAGNOSTICO MODELO V2")
print("==========================================")

print(
    f"\nModelo seleccionado en epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation Loss: "
    f"{checkpoint['val_loss']:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)

print("\nClases:")
print(checkpoint["classes"])


# ============================================================
# EVALUACION
# ============================================================

def evaluar(loader, dataset, nombre):

    num_classes = len(
        dataset.classes
    )

    matriz = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64
    )

    correctas = 0
    total = 0

    with torch.no_grad():

        for imagenes, etiquetas in loader:

            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            salidas = modelo(
                imagenes
            )

            predicciones = torch.argmax(
                salidas,
                dim=1
            )

            correctas += (
                predicciones == etiquetas
            ).sum().item()

            total += etiquetas.size(0)

            for real, pred in zip(
                etiquetas.cpu(),
                predicciones.cpu()
            ):

                matriz[
                    real.item(),
                    pred.item()
                ] += 1


    accuracy = (
        100.0
        * correctas
        / total
    )


    print("\n==========================================")
    print(nombre)
    print("==========================================")

    print(
        f"\nAccuracy total: "
        f"{accuracy:.2f}%"
    )

    print("\nAccuracy por clase:")

    for i, clase in enumerate(
        dataset.classes
    ):

        total_clase = (
            matriz[i].sum().item()
        )

        correctas_clase = (
            matriz[i, i].item()
        )

        accuracy_clase = (
            100.0
            * correctas_clase
            / total_clase
        )

        print(
            f"Clase {clase}: "
            f"{correctas_clase}/"
            f"{total_clase} "
            f"= {accuracy_clase:.2f}%"
        )


    print("\nMATRIZ DE CONFUSION")

    print(
        "Filas = clase real"
    )

    print(
        "Columnas = prediccion\n"
    )

    print(matriz)


# ============================================================
# TRAIN
# ============================================================

evaluar(
    train_loader,
    train_dataset,
    "TRAIN SIN DATA AUGMENTATION"
)


# ============================================================
# VALIDATION
# ============================================================

evaluar(
    val_loader,
    val_dataset,
    "VALIDATION"
)


print("\n==========================================")
print("DIAGNOSTICO FINALIZADO")
print("==========================================")

print(
    "\nEl conjunto TEST NO fue utilizado."
)