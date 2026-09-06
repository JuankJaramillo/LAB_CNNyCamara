import os
import random
import matplotlib.pyplot as plt
from PIL import Image


PERSONA = "persona_01"

SESIONES = [
    "sesion_01",
    "sesion_02",
    "sesion_03",
    "sesion_04"
]

CLASES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

DATASET_RAW = os.path.join(
    "dataset_raw",
    PERSONA
)

random.seed(42)


plt.figure(
    figsize=(15, 11)
)


posicion = 1


for sesion in SESIONES:

    for clase in CLASES:

        carpeta = os.path.join(
            DATASET_RAW,
            sesion,
            clase
        )

        archivos = [
            archivo
            for archivo in os.listdir(carpeta)
            if archivo.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        archivo = random.choice(
            archivos
        )

        ruta = os.path.join(
            carpeta,
            archivo
        )

        imagen = Image.open(
            ruta
        ).convert("RGB")

        plt.subplot(
            len(SESIONES),
            len(CLASES),
            posicion
        )

        plt.imshow(
            imagen
        )

        if sesion == SESIONES[0]:

            plt.title(
                f"Clase {clase}\n{clase} dedos"
            )

        if clase == "0":

            plt.ylabel(
                sesion,
                fontsize=11
            )

        plt.xticks([])
        plt.yticks([])

        posicion += 1


plt.suptitle(
    "Comparacion de clases entre sesiones",
    fontsize=16
)

plt.tight_layout()

plt.show()