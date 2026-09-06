import os
import shutil

ORIGEN = "dataset_raw/persona_01"
DESTINO = "dataset_v2"

CLASES = ["0", "1", "2", "3", "4"]

PARTICIONES = {
    "train": [
        "sesion_01",
        "sesion_02",
        "sesion_03",
        "sesion_04"
    ],

    "val": [
        "sesion_05"
    ],

    "test": [
        "sesion_06"
    ]
}


# ============================================================
# ELIMINAR DATASET V2 ANTERIOR SI EXISTE
# ============================================================

if os.path.exists(DESTINO):
    shutil.rmtree(DESTINO)


# ============================================================
# CREAR PARTICIONES
# ============================================================

for particion, sesiones in PARTICIONES.items():

    for clase in CLASES:

        ruta_destino = os.path.join(
            DESTINO,
            particion,
            clase
        )

        os.makedirs(
            ruta_destino,
            exist_ok=True
        )

        for sesion in sesiones:

            ruta_origen = os.path.join(
                ORIGEN,
                sesion,
                clase
            )

            if not os.path.exists(ruta_origen):

                print(
                    f"ADVERTENCIA: no existe "
                    f"{ruta_origen}"
                )

                continue

            archivos = [
                archivo
                for archivo in os.listdir(ruta_origen)
                if archivo.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            for archivo in archivos:

                origen_archivo = os.path.join(
                    ruta_origen,
                    archivo
                )

                destino_archivo = os.path.join(
                    ruta_destino,
                    archivo
                )

                shutil.copy2(
                    origen_archivo,
                    destino_archivo
                )


# ============================================================
# MOSTRAR RESUMEN
# ============================================================

print("\n==========================================")
print("DATASET V2 - PARTICION POR SESIONES")
print("==========================================")

print("\nTRAIN:")
print("Sesiones 01, 02, 03 y 04")

print("\nVALIDATION:")
print("Sesion 05")

print("\nTEST FINAL:")
print("Sesion 06")


total_general = 0


for particion in [
    "train",
    "val",
    "test"
]:

    print("\n==========================================")
    print(particion.upper())
    print("==========================================")

    total_particion = 0

    for clase in CLASES:

        ruta = os.path.join(
            DESTINO,
            particion,
            clase
        )

        archivos = [
            archivo
            for archivo in os.listdir(ruta)
            if archivo.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        cantidad = len(
            archivos
        )

        total_particion += cantidad

        print(
            f"Clase {clase}: "
            f"{cantidad} imagenes"
        )

    print(
        f"Total {particion}: "
        f"{total_particion} imagenes"
    )

    total_general += total_particion


print("\n==========================================")
print(
    f"TOTAL DATASET V2: "
    f"{total_general} imagenes"
)
print("==========================================")

print("\nDataset V1 original conservado.")
print("Dataset V2 creado en: dataset_v2/")