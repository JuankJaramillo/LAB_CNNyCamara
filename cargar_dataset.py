import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from transforms_cnn import train_transform, eval_transform


BATCH_SIZE = 32


# ============================================================
# CARGAR DATASETS
# ============================================================

train_dataset = ImageFolder(
    root="dataset/train",
    transform=train_transform
)

val_dataset = ImageFolder(
    root="dataset/val",
    transform=eval_transform
)

test_dataset = ImageFolder(
    root="dataset/test",
    transform=eval_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# INFORMACION GENERAL
# ============================================================

print("==========================================")
print("DATASET CNN")
print("==========================================")

print("\nClases detectadas:")
print(train_dataset.classes)

print("\nCorrespondencia clase -> indice:")
print(train_dataset.class_to_idx)

print("\nCantidad de imagenes:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

print(
    f"\nTotal: "
    f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
)


# ============================================================
# CANTIDAD POR CLASE
# ============================================================

print("\n==========================================")
print("DISTRIBUCION POR CLASE")
print("==========================================")

for nombre, dataset in [
    ("TRAIN", train_dataset),
    ("VALIDATION", val_dataset),
    ("TEST", test_dataset)
]:

    print(f"\n{nombre}")

    conteo = {
        clase: 0
        for clase in dataset.classes
    }

    for _, etiqueta in dataset.samples:

        clase = dataset.classes[etiqueta]

        conteo[clase] += 1

    for clase, cantidad in conteo.items():

        print(
            f"Clase {clase}: {cantidad} imagenes"
        )


# ============================================================
# PROBAR UN BATCH
# ============================================================

imagenes, etiquetas = next(
    iter(train_loader)
)

print("\n==========================================")
print("PRUEBA DE BATCH")
print("==========================================")

print("\nForma del batch de imagenes:")
print(imagenes.shape)

print("\nForma del batch de etiquetas:")
print(etiquetas.shape)

print("\nEtiquetas del primer batch:")
print(etiquetas)

print("\nBatch size:")
print(BATCH_SIZE)

print("\nCarga del dataset completada correctamente.")