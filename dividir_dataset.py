import os
import shutil

ORIGEN = "dataset_raw/persona_01"
DESTINO = "dataset"

CLASES = ["0", "1", "2", "3", "4"]

PARTICIONES = {
    "train": ["sesion_02", "sesion_03"],
    "val": ["sesion_01"],
    "test": ["sesion_04"]
}

# Eliminar dataset anterior si existe
if os.path.exists(DESTINO):
    shutil.rmtree(DESTINO)

# Crear estructura y copiar archivos
for particion, sesiones in PARTICIONES.items():

    for clase in CLASES:

        ruta_destino = os.path.join(
            DESTINO,
            particion,
            clase
        )

        os.makedirs(ruta_destino, exist_ok=True)

        for sesion in sesiones:

            ruta_origen = os.path.join(
                ORIGEN,
                sesion,
                clase
            )

            if not os.path.exists(ruta_origen):
                print(f"ADVERTENCIA: no existe {ruta_origen}")
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

# Mostrar resumen
print("\n========================================")
print("PARTICION DEL DATASET")
print("========================================")

total_general = 0

for particion in ["train", "val", "test"]:

    print(f"\n{particion.upper()}")

    total_particion = 0

    for clase in CLASES:

        ruta = os.path.join(
            DESTINO,
            particion,
            clase
        )

        cantidad = len([
            archivo
            for archivo in os.listdir(ruta)
            if archivo.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        total_particion += cantidad

        print(
            f"Clase {clase}: {cantidad} imagenes"
        )

    print(
        f"Total {particion}: {total_particion} imagenes"
    )

    total_general += total_particion

print("\n========================================")
print(f"TOTAL DATASET: {total_general} imagenes")
print("========================================")