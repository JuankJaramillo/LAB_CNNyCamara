import os
import random
import matplotlib.pyplot as plt
from PIL import Image


PERSONA = "persona_01"

SESIONES = [
    "sesion_01",
    "sesion_02",
    "sesion_03",
    "sesion_04",
    "sesion_05"
]

CLASES = [
    "2",
    "3",
    "4"
]

MUESTRAS_POR_SESION = 2

DATASET_RAW = os.path.join(
    "dataset_raw",
    PERSONA
)

random.seed(42)


# ============================================================
# FIGURA
# ============================================================

columnas = (
    len(SESIONES)
    * MUESTRAS_POR_SESION
)

filas = len(CLASES)

plt.figure(
    figsize=(20, 8)
)

posicion = 1


for clase in CLASES:

    for sesion in SESIONES:

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

        muestras = random.sample(
            archivos,
            min(
                MUESTRAS_POR_SESION,
                len(archivos)
            )
        )

        for indice, archivo in enumerate(
            muestras
        ):

            ruta = os.path.join(
                carpeta,
                archivo
            )

            imagen = Image.open(
                ruta
            ).convert("RGB")

            plt.subplot(
                filas,
                columnas,
                posicion
            )

            plt.imshow(
                imagen
            )

            if clase == CLASES[0]:

                plt.title(
                    f"{sesion}\n"
                    f"M{indice + 1}",
                    fontsize=8
                )

            if posicion % columnas == 1:

                plt.ylabel(
                    f"Clase {clase}",
                    fontsize=12
                )

            plt.xticks([])
            plt.yticks([])

            posicion += 1


plt.suptitle(
    "Comparacion de clases problematicas entre sesiones",
    fontsize=16
)

plt.tight_layout()

plt.show()