import os
import cv2

from landmark_input import (
    crear_detector_landmarks,
    crear_imagen_landmarks
)


# ============================================================
# RUTAS
# ============================================================

ORIGEN = "dataset_v2/test"
DESTINO = "dataset_landmarks/test"

CLASES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]


# ============================================================
# CREAR DIRECTORIOS
# ============================================================

os.makedirs(
    DESTINO,
    exist_ok=True
)

for clase in CLASES:

    os.makedirs(
        os.path.join(
            DESTINO,
            clase
        ),
        exist_ok=True
    )


# ============================================================
# MEDIAPIPE
# ============================================================

detector = crear_detector_landmarks(
    static_image_mode=True
)


# ============================================================
# CONTADORES
# ============================================================

total = 0
generadas = 0
fallos = 0

estadisticas = {}


# ============================================================
# PROCESAR TEST
# ============================================================

print("==========================================")
print("CREACION DEL TEST DE LANDMARKS")
print("==========================================")


for clase in CLASES:

    carpeta_origen = os.path.join(
        ORIGEN,
        clase
    )

    carpeta_destino = os.path.join(
        DESTINO,
        clase
    )

    archivos = sorted(
        os.listdir(
            carpeta_origen
        )
    )

    clase_total = 0
    clase_generadas = 0
    clase_fallos = 0


    for archivo in archivos:

        ruta = os.path.join(
            carpeta_origen,
            archivo
        )


        imagen = cv2.imread(
            ruta
        )


        if imagen is None:

            continue


        total += 1
        clase_total += 1


        imagen_landmarks, bbox = (
            crear_imagen_landmarks(
                imagen,
                detector,
                size=128
            )
        )


        if imagen_landmarks is None:

            fallos += 1
            clase_fallos += 1

            continue


        nombre_salida = (
            os.path.splitext(
                archivo
            )[0]
            + ".png"
        )


        ruta_salida = os.path.join(
            carpeta_destino,
            nombre_salida
        )


        cv2.imwrite(
            ruta_salida,
            imagen_landmarks
        )


        generadas += 1
        clase_generadas += 1


    estadisticas[clase] = {
        "total": clase_total,
        "generadas": clase_generadas,
        "fallos": clase_fallos
    }


# ============================================================
# CERRAR DETECTOR
# ============================================================

detector.close()


# ============================================================
# RESULTADOS
# ============================================================

print()

for clase in CLASES:

    datos = estadisticas[
        clase
    ]

    print(
        f"Clase {clase}: "
        f"total={datos['total']} | "
        f"generadas={datos['generadas']} | "
        f"fallos={datos['fallos']}"
    )


print()
print("------------------------------------------")

print(
    f"Total procesadas: {total}"
)

print(
    f"Landmarks generados: {generadas}"
)

print(
    f"Fallos deteccion: {fallos}"
)


if total > 0:

    tasa = (
        generadas
        / total
        * 100.0
    )

    print(
        f"Tasa deteccion: {tasa:.2f}%"
    )


print("==========================================")
print("TEST PREPARADO")
print("==========================================")