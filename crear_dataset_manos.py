import os
import cv2
import shutil

from hand_crop import (
    crear_detector,
    extraer_mano
)


# ============================================================
# CONFIGURACION
# ============================================================

ORIGEN = "dataset_v2"

DESTINO = "dataset_manos"

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

TAMANO_SALIDA = 256


# ============================================================
# RECREAR DATASET
# ============================================================

if os.path.exists(DESTINO):
    shutil.rmtree(DESTINO)

for particion in PARTICIONES:

    for clase in CLASES:

        os.makedirs(
            os.path.join(
                DESTINO,
                particion,
                clase
            ),
            exist_ok=True
        )


# ============================================================
# MEDIAPIPE PARA IMAGENES ESTATICAS
# ============================================================

detector = crear_detector(
    static_image_mode=True
)


# ============================================================
# CONTADORES
# ============================================================

total_procesadas = 0
total_detectadas = 0
total_fallidas = 0


print("==========================================")
print("CREACION DATASET CENTRADO EN MANOS")
print("==========================================")


# ============================================================
# PROCESAMIENTO
# ============================================================

for particion in PARTICIONES:

    print(
        f"\n=========================================="
    )

    print(
        particion.upper()
    )

    print(
        "=========================================="
    )

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

            recorte, bbox = extraer_mano(
                imagen,
                detector,
                margen=0.35
            )

            if recorte is None:

                fallidas_clase += 1
                total_fallidas += 1

                continue


            # =================================================
            # UNIFORMIZAR A 256 x 256
            # =================================================

            recorte = cv2.resize(
                recorte,
                (
                    TAMANO_SALIDA,
                    TAMANO_SALIDA
                )
            )


            # =================================================
            # GUARDAR
            # =================================================

            ruta_destino = os.path.join(
                carpeta_destino,
                archivo
            )

            cv2.imwrite(
                ruta_destino,
                recorte
            )

            detectadas_clase += 1
            total_detectadas += 1


        print(
            f"Clase {clase}: "
            f"{detectadas_clase} detectadas | "
            f"{fallidas_clase} fallidas"
        )


# ============================================================
# FINALIZAR
# ============================================================

detector.close()


print("\n==========================================")
print("RESUMEN")
print("==========================================")

print(
    f"Imagenes procesadas: "
    f"{total_procesadas}"
)

print(
    f"Manos detectadas: "
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

print(
    "\nDataset generado en:"
)

print(
    DESTINO
)