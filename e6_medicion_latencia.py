import os
import csv
import time

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from landmark_input import (
    crear_detector_landmarks,
    crear_imagen_landmarks
)

from modelo_cnn_v3 import FingerCNNV3

from transforms_cnn_v3 import (
    eval_transform_v3
)

from command_filter import (
    CommandFilter
)

from robot_controller import (
    RobotController
)


# ============================================================
# CONFIGURACION
# ============================================================

MODEL_PATH = "models_v3/best_cnn_v3.pth"

RESULTS_PATH = "results_e6"

os.makedirs(
    RESULTS_PATH,
    exist_ok=True
)


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CARGAR CNN
# ============================================================

modelo = FingerCNNV3(
    num_classes=5
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

modelo.load_state_dict(
    checkpoint["model_state_dict"]
)

modelo = modelo.to(
    device
)

modelo.eval()


print("==========================================")
print("E6 - MEDICION DE LATENCIA")
print("==========================================")

print()
print(
    f"Modelo: epoch {checkpoint['epoch']}"
)

print(
    f"Val accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)

print(
    f"Dispositivo: {device}"
)


# ============================================================
# MEDIAPIPE
# ============================================================

detector = crear_detector_landmarks(
    static_image_mode=False
)


# ============================================================
# FILTRO
# ============================================================

filtro = CommandFilter(
    window_size=7,
    min_votes=5,
    confidence_threshold=0.80,
    cooldown=0.8
)


# ============================================================
# ROBOT
# ============================================================

robot = RobotController()


# ============================================================
# CAMARA
# ============================================================

camara = cv2.VideoCapture(
    0
)

if not camara.isOpened():

    robot.cerrar()
    detector.close()

    raise RuntimeError(
        "No se pudo abrir la camara."
    )


# ============================================================
# LISTAS DE RESULTADOS
# ============================================================

frames_resultados = []

comandos_resultados = []

contador_frame = 0
contador_comando = 0


ultimo_comando = "NINGUNO"

tiempo_ultimo_comando = 0.0


# ============================================================
# FUNCION PERCENTILES
# ============================================================

def estadisticas(lista):

    if len(lista) == 0:

        return {
            "n": 0,
            "media": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan
        }

    datos = np.array(
        lista,
        dtype=float
    )

    return {
        "n": len(datos),

        "media": np.mean(
            datos
        ),

        "p50": np.percentile(
            datos,
            50
        ),

        "p95": np.percentile(
            datos,
            95
        ),

        "min": np.min(
            datos
        ),

        "max": np.max(
            datos
        )
    }


# ============================================================
# INSTRUCCIONES
# ============================================================

print()
print("==========================================")
print("PRUEBA EN VIVO")
print("==========================================")

print()
print("Realiza varias veces:")
print()
print("0 -> 1 -> 0")
print("0 -> 2 -> 0")
print("0 -> 3 -> 0")
print("0 -> 4 -> 0")

print()
print(
    "Para J1-J3 alterna izquierda/derecha "
    "para no llegar a los limites."
)

print()
print(
    "Intenta generar al menos 5-10 comandos "
    "de cada clase."
)

print()
print(
    "Presiona Q para terminar y generar resultados."
)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:

    while True:

        contador_frame += 1

        # ====================================================
        # CAPTURA
        # ====================================================

        ret, frame = camara.read()

        if not ret:
            break

        frame = cv2.flip(
            frame,
            1
        )

        alto, ancho, _ = frame.shape


        # ====================================================
        # VARIABLES DEL FRAME
        # ====================================================

        clase_int = -1
        confianza = 0.0

        landmarks_ms = np.nan
        cnn_ms = np.nan
        percepcion_ms = np.nan
        filtro_ms = np.nan

        comando = None

        direccion = 0
        direccion_texto = "-"


        # ====================================================
        # MEDIR MEDIAPIPE / LANDMARKS
        # ====================================================

        t0 = time.perf_counter()

        imagen_landmarks, bbox = (
            crear_imagen_landmarks(
                frame,
                detector,
                size=128
            )
        )

        t1 = time.perf_counter()

        landmarks_ms = (
            t1 - t0
        ) * 1000.0


        # ====================================================
        # SI HAY MANO
        # ====================================================

        if imagen_landmarks is not None:

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # DIRECCION
            # =================================================

            centro_x = (
                x1 + x2
            ) // 2

            centro_pantalla = (
                ancho // 2
            )


            if centro_x < centro_pantalla:

                direccion = -1

                direccion_texto = (
                    "IZQUIERDA (-)"
                )

            else:

                direccion = 1

                direccion_texto = (
                    "DERECHA (+)"
                )


            cv2.line(
                frame,
                (centro_pantalla, 0),
                (centro_pantalla, alto),
                (255, 0, 0),
                1
            )


            # =================================================
            # PREPARACION CNN
            # =================================================

            imagen_rgb = cv2.cvtColor(
                imagen_landmarks,
                cv2.COLOR_BGR2RGB
            )

            imagen_pil = Image.fromarray(
                imagen_rgb
            )

            tensor = eval_transform_v3(
                imagen_pil
            )

            tensor = tensor.unsqueeze(
                0
            ).to(
                device
            )


            # =================================================
            # MEDIR CNN
            # =================================================

            t2 = time.perf_counter()

            with torch.no_grad():

                logits = modelo(
                    tensor
                )

                probabilidades = torch.softmax(
                    logits,
                    dim=1
                )

                confianza_tensor, prediccion = (
                    torch.max(
                        probabilidades,
                        dim=1
                    )
                )

            t3 = time.perf_counter()


            cnn_ms = (
                t3 - t2
            ) * 1000.0


            clase_int = (
                prediccion.item()
            )

            confianza = (
                confianza_tensor.item()
            )


            # =================================================
            # PERCEPCION TOTAL
            # =================================================

            percepcion_ms = (
                landmarks_ms
                + cnn_ms
            )


            # =================================================
            # MEDIR FILTRO
            # =================================================

            tf0 = time.perf_counter()

            comando = filtro.update(
                clase_int,
                confianza
            )

            tf1 = time.perf_counter()


            filtro_ms = (
                tf1 - tf0
            ) * 1000.0


            # =================================================
            # GUARDAR FRAME
            # =================================================

            frames_resultados.append(
                {
                    "frame": contador_frame,
                    "clase": clase_int,
                    "confianza": confianza * 100.0,
                    "landmarks_ms": landmarks_ms,
                    "cnn_ms": cnn_ms,
                    "percepcion_ms": percepcion_ms,
                    "filtro_ms": filtro_ms
                }
            )


            # =================================================
            # COMANDO ESTABLE
            # =================================================

            if comando is not None:

                contador_comando += 1

                ejecucion_ms = np.nan

                tipo = ""

                accion = ""


                # =============================================
                # J1
                # =============================================

                if comando == 1:

                    tipo = "joint"

                    accion = (
                        f"J1_{direccion_texto}"
                    )

                    te0 = time.perf_counter()

                    robot.mover_joint(
                        1,
                        direccion
                    )

                    te1 = time.perf_counter()

                    ejecucion_ms = (
                        te1 - te0
                    ) * 1000.0


                # =============================================
                # J2
                # =============================================

                elif comando == 2:

                    tipo = "joint"

                    accion = (
                        f"J2_{direccion_texto}"
                    )

                    te0 = time.perf_counter()

                    robot.mover_joint(
                        2,
                        direccion
                    )

                    te1 = time.perf_counter()

                    ejecucion_ms = (
                        te1 - te0
                    ) * 1000.0


                # =============================================
                # J3
                # =============================================

                elif comando == 3:

                    tipo = "joint"

                    accion = (
                        f"J3_{direccion_texto}"
                    )

                    te0 = time.perf_counter()

                    robot.mover_joint(
                        3,
                        direccion
                    )

                    te1 = time.perf_counter()

                    ejecucion_ms = (
                        te1 - te0
                    ) * 1000.0


                # =============================================
                # GRIPPER
                # =============================================

                elif comando == 4:

                    tipo = "gripper"

                    accion = "GRIPPER"

                    te0 = time.perf_counter()

                    robot.toggle_gripper()

                    te1 = time.perf_counter()

                    ejecucion_ms = (
                        te1 - te0
                    ) * 1000.0


                comandos_resultados.append(
                    {
                        "comando_num": contador_comando,
                        "clase": comando,
                        "tipo": tipo,
                        "accion": accion,
                        "ejecucion_ms": ejecucion_ms
                    }
                )


                ultimo_comando = accion

                tiempo_ultimo_comando = (
                    time.time()
                )


                print()
                print(
                    "LATENCIA EJECUCION:"
                )

                print(
                    f"{accion} -> "
                    f"{ejecucion_ms:.2f} ms"
                )


            # =================================================
            # ENTRADA CNN
            # =================================================

            entrada_grande = cv2.resize(
                imagen_landmarks,
                (300, 300),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow(
                "Entrada CNN - E6",
                entrada_grande
            )


        # ====================================================
        # SIN MANO
        # ====================================================

        else:

            filtro.reset()


        # ====================================================
        # INTERFAZ
        # ====================================================

        cv2.putText(
            frame,
            "E6 - MEDICION DE LATENCIA",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


        if clase_int >= 0:

            cv2.putText(
                frame,
                f"Clase: {clase_int}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confianza: {confianza * 100:.1f}%",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Percepcion: {percepcion_ms:.1f} ms",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Filtro: {filtro_ms:.3f} ms",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )


        if (
            time.time()
            - tiempo_ultimo_comando
            < 1.5
        ):

            cv2.putText(
                frame,
                f"COMANDO: {ultimo_comando}",
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


        cv2.putText(
            frame,
            f"Comandos medidos: {contador_comando}",
            (20, alto - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Q: finalizar",
            (20, alto - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "E6 - Latencia sistema",
            frame
        )


        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        if tecla == ord("q"):
            break


# ============================================================
# FINALIZACION
# ============================================================

finally:

    camara.release()

    detector.close()

    cv2.destroyAllWindows()

    robot.cerrar()


# ============================================================
# EXTRAER LATENCIAS
# ============================================================

landmarks_lista = [
    x["landmarks_ms"]
    for x in frames_resultados
]

cnn_lista = [
    x["cnn_ms"]
    for x in frames_resultados
]

percepcion_lista = [
    x["percepcion_ms"]
    for x in frames_resultados
]

filtro_lista = [
    x["filtro_ms"]
    for x in frames_resultados
]


ejecucion_joint_lista = [
    x["ejecucion_ms"]
    for x in comandos_resultados
    if x["tipo"] == "joint"
]


ejecucion_gripper_lista = [
    x["ejecucion_ms"]
    for x in comandos_resultados
    if x["tipo"] == "gripper"
]


# ============================================================
# CALCULAR ESTADISTICAS
# ============================================================

resumen = {

    "Landmarks": estadisticas(
        landmarks_lista
    ),

    "CNN": estadisticas(
        cnn_lista
    ),

    "Percepcion total": estadisticas(
        percepcion_lista
    ),

    "Filtro": estadisticas(
        filtro_lista
    ),

    "Ejecucion joints": estadisticas(
        ejecucion_joint_lista
    ),

    "Comando gripper": estadisticas(
        ejecucion_gripper_lista
    )
}


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print()
print("==========================================")
print("RESULTADOS E6 - LATENCIA")
print("==========================================")

print()

print(
    f"{'Componente':<22}"
    f"{'N':<8}"
    f"{'Media':<12}"
    f"{'p50':<12}"
    f"{'p95':<12}"
)


for nombre, datos in resumen.items():

    print(
        f"{nombre:<22}"
        f"{datos['n']:<8}"
        f"{datos['media']:<12.3f}"
        f"{datos['p50']:<12.3f}"
        f"{datos['p95']:<12.3f}"
    )


# ============================================================
# CSV FRAMES
# ============================================================

ruta_frames = os.path.join(
    RESULTS_PATH,
    "latencias_frames.csv"
)


with open(
    ruta_frames,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    campos = [
        "frame",
        "clase",
        "confianza",
        "landmarks_ms",
        "cnn_ms",
        "percepcion_ms",
        "filtro_ms"
    ]

    writer = csv.DictWriter(
        archivo,
        fieldnames=campos
    )

    writer.writeheader()

    writer.writerows(
        frames_resultados
    )


# ============================================================
# CSV COMANDOS
# ============================================================

ruta_comandos = os.path.join(
    RESULTS_PATH,
    "latencias_comandos.csv"
)


with open(
    ruta_comandos,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    campos = [
        "comando_num",
        "clase",
        "tipo",
        "accion",
        "ejecucion_ms"
    ]

    writer = csv.DictWriter(
        archivo,
        fieldnames=campos
    )

    writer.writeheader()

    writer.writerows(
        comandos_resultados
    )


# ============================================================
# TXT RESUMEN
# ============================================================

ruta_resumen = os.path.join(
    RESULTS_PATH,
    "resumen_latencia.txt"
)


with open(
    ruta_resumen,
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(
        "E6 - LATENCIA DEL SISTEMA\n"
    )

    archivo.write(
        "=========================\n\n"
    )


    for nombre, datos in resumen.items():

        archivo.write(
            f"{nombre}\n"
        )

        archivo.write(
            f"N = {datos['n']}\n"
        )

        archivo.write(
            f"Media = {datos['media']:.3f} ms\n"
        )

        archivo.write(
            f"p50 = {datos['p50']:.3f} ms\n"
        )

        archivo.write(
            f"p95 = {datos['p95']:.3f} ms\n\n"
        )


# ============================================================
# GRAFICA p50 / p95
# ============================================================

componentes = [
    "Percepcion",
    "Filtro",
    "Ejecucion joints"
]

p50 = [
    resumen["Percepcion total"]["p50"],
    resumen["Filtro"]["p50"],
    resumen["Ejecucion joints"]["p50"]
]

p95 = [
    resumen["Percepcion total"]["p95"],
    resumen["Filtro"]["p95"],
    resumen["Ejecucion joints"]["p95"]
]


x = np.arange(
    len(componentes)
)

ancho = 0.35


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.bar(
    x - ancho / 2,
    p50,
    ancho,
    label="p50"
)

ax.bar(
    x + ancho / 2,
    p95,
    ancho,
    label="p95"
)


ax.set_ylabel(
    "Latencia [ms]"
)

ax.set_title(
    "E6 - Latencia por componente"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    componentes
)

ax.legend()


plt.tight_layout()


ruta_grafica = os.path.join(
    RESULTS_PATH,
    "latencia_p50_p95.png"
)


plt.savefig(
    ruta_grafica,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("==========================================")
print("ARCHIVOS GENERADOS")
print("==========================================")

print(
    ruta_frames
)

print(
    ruta_comandos
)

print(
    ruta_resumen
)

print(
    ruta_grafica
)

print()
print("==========================================")
print("E6-A COMPLETADO")
print("==========================================")
