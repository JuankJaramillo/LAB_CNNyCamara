import os
import matplotlib.pyplot as plt
from PIL import Image

from transforms_cnn import train_transform

CLASE = "2"

CARPETA = os.path.join(
    "dataset",
    "train",
    CLASE
)

archivos = [
    archivo
    for archivo in os.listdir(CARPETA)
    if archivo.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

if len(archivos) == 0:
    print("ERROR: No se encontraron imagenes.")
    exit()

ruta_imagen = os.path.join(
    CARPETA,
    archivos[0]
)

imagen_original = Image.open(
    ruta_imagen
).convert("RGB")


def desnormalizar(tensor):

    imagen = tensor.permute(
        1, 2, 0
    ).numpy()

    imagen = imagen * 0.5 + 0.5

    imagen = imagen.clip(
        0, 1
    )

    return imagen


plt.figure(figsize=(12, 7))

plt.subplot(2, 3, 1)

plt.imshow(imagen_original)

plt.title(
    f"Original - Clase {CLASE}"
)

plt.axis("off")

for i in range(5):

    imagen_transformada = train_transform(
        imagen_original
    )

    imagen_transformada = desnormalizar(
        imagen_transformada
    )

    plt.subplot(
        2,
        3,
        i + 2
    )

    plt.imshow(
        imagen_transformada
    )

    plt.title(
        f"Aumento {i + 1}"
    )

    plt.axis("off")

plt.suptitle(
    "Pipeline de Data Augmentation - CNN"
)

plt.tight_layout()

plt.show()  