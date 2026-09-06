import os
import cv2
import shutil

from landmark_input import (
    crear_detector_landmarks,
    crear_imagen_landmarks
)


# ============================================================
# CONFIGURACION
# ============================================================

ORIGEN = "dataset_v2"
DESTINO = "dataset_landmarks"

PARTICIONES = [
    "train",
    "val"
]

CLASES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]


# ============================================================
# RECREAR DATASET
# ============================================================

if os.path.exists(DESTINO):
    shutil.rmtree(DESTINO)


for particion in PARTICIONES:

    for clase in CLASES:

        ruta = os.path.join(
            DESTINO,
            particion,
            clase
        )

        os.makedirs(
            ruta,
            exist_ok=True
        )


# ============================================================
# DETECTOR PARA IMAGENES ESTATICAS
# ============================================================

detector = crear_detector_landmarks(
    static_image_mode=True
)


# ============================================================
# CONTADORES
# ============================================================

total_procesadas = 0
total_detectadas = 0
total_fallidas = 0


print("==========================================")
print("CREACION DATASET DE LANDMARKS")
print("==========================================")


# ============================================================
# PROCESAR TRAIN Y VALIDATION
# ============================================================

for particion in PARTICIONES:

    print()
    print("==========================================")
    print(particion.upper())
    print("==========================================")

    for clase in CLASES:

        carpeta_origen = os.path.join(
            ORIGEN,
            particion,
            clase
        )

        carpeta_destino = os.path.join(
            DESTINO,
            particion,
            clase
        )

        archivos = [
            archivo
            for archivo in os.listdir(
                carpeta_origen
            )
            if archivo.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ]

        detectadas_clase = 0
        fallidas_clase = 0

        for archivo in archivos:

            total_procesadas += 1

            ruta_origen = os.path.join(
                carpeta_origen,
                archivo
            )

            imagen = cv2.imread(
                ruta_origen
            )

            if imagen is None:

                fallidas_clase += 1
                total_fallidas += 1

                continue


            imagen_landmarks, bbox = (
                crear_imagen_landmarks(
                    imagen,
                    detector,
                    size=128
                )
            )


            if imagen_landmarks is None:

                fallidas_clase += 1
                total_fallidas += 1

                continue


            ruta_destino = os.path.join(
                carpeta_destino,
                archivo
            )


            cv2.imwrite(
                ruta_destino,
                imagen_landmarks
            )


            detectadas_clase += 1
            total_detectadas += 1


        print(
            f"Clase {clase}: "
            f"{detectadas_clase} generadas | "
            f"{fallidas_clase} fallidas"
        )


# ============================================================
# CERRAR DETECTOR
# ============================================================

detector.close()


# ============================================================
# RESUMEN
# ============================================================

print()
print("==========================================")
print("RESUMEN")
print("==========================================")

print(
    f"Imagenes procesadas: "
    f"{total_procesadas}"
)

print(
    f"Landmarks generados: "
    f"{total_detectadas}"
)

print(
    f"Detecciones fallidas: "
    f"{total_fallidas}"
)


if total_procesadas > 0:

    porcentaje = (
        100.0
        * total_detectadas
        / total_procesadas
    )

    print(
        f"Tasa de deteccion: "
        f"{porcentaje:.2f}%"
    )


print()
print(
    "Dataset generado en:"
)

print(
    DESTINO
)