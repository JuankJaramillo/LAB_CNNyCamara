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

from modelo_cnn_v3 import (
    FingerCNNV3
)

from transforms_cnn_v3 import (
    eval_transform_v3
)

from command_filter import (
    CommandFilter
)


# ============================================================
# CONFIGURACION
# ============================================================

MODEL_PATH = "models_v3/best_cnn_v3.pth"

RESULTS_PATH = "results_e6"

DURACION_ENSAYO = 4.0

CLASES = [
    0,
    1,
    2,
    3,
    4
]

CONDICIONES = [
    "normal",
    "lejos",
    "lateral",
    "baja_luz"
]


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
# CARGAR MODELO
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
print("E6-B - PRUEBA DE ROBUSTEZ")
print("==========================================")

print()
print(
    f"Modelo: epoch {checkpoint['epoch']}"
)

print(
    f"Validation Accuracy: "
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
# CAMARA
# ============================================================

camara = cv2.VideoCapture(
    0
)


if not camara.isOpened():

    detector.close()

    raise RuntimeError(
        "No se pudo abrir la camara."
    )


# ============================================================
# WARM-UP CAMARA
# ============================================================

print()
print(
    "Inicializando camara..."
)


for _ in range(20):

    camara.read()


# ============================================================
# RESULTADOS
# ============================================================

resultados_ensayos = []

predicciones_frames = []


# ============================================================
# DESCRIPCIONES
# ============================================================

descripciones = {

    "normal":
        "Iluminacion normal, mano centrada y distancia habitual.",

    "lejos":
        "Aleja la mano aproximadamente a 1 - 1.2 metros.",

    "lateral":
        "Ubica la mano cerca de un borde de la imagen.",

    "baja_luz":
        "Reduce claramente la iluminacion del entorno."
}


# ============================================================
# EJECUTAR ENSAYO
# ============================================================

def ejecutar_ensayo(
    condicion,
    clase_esperada,
    numero_ensayo
):

    filtro = CommandFilter(
        window_size=7,
        min_votes=5,
        confidence_threshold=0.80,
        cooldown=0.8
    )


    # --------------------------------------------------------
    # CONTADORES
    # --------------------------------------------------------

    frames_totales = 0

    frames_detectados = 0

    frames_correctos = 0

    suma_confianza = 0.0


    comandos_emitidos = 0

    comandos_correctos = 0

    falsos_comandos = 0


    comandos_obtenidos = []


    # --------------------------------------------------------
    # PREPARACION
    # --------------------------------------------------------

    print()
    print("==========================================")

    print(
        f"ENSAYO {numero_ensayo}"
    )

    print(
        f"Condicion: {condicion}"
    )

    print(
        f"Clase esperada: {clase_esperada}"
    )

    print("------------------------------------------")

    print(
        descripciones[
            condicion
        ]
    )

    print()

    input(
        "Prepara el gesto y presiona ENTER "
        "para comenzar..."
    )


    filtro.reset()


    tiempo_inicio = time.perf_counter()


    # --------------------------------------------------------
    # BUCLE DEL ENSAYO
    # --------------------------------------------------------

    while (
        time.perf_counter()
        - tiempo_inicio
        < DURACION_ENSAYO
    ):

        ret, frame = camara.read()


        if not ret:

            continue


        frame = cv2.flip(
            frame,
            1
        )


        frames_totales += 1


        imagen_landmarks, bbox = (
            crear_imagen_landmarks(
                frame,
                detector,
                size=128
            )
        )


        clase_predicha = -1

        confianza = 0.0

        comando = None


        # ====================================================
        # MANO DETECTADA
        # ====================================================

        if imagen_landmarks is not None:

            frames_detectados += 1


            # ------------------------------------------------
            # PREPARAR CNN
            # ------------------------------------------------

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


            # ------------------------------------------------
            # INFERENCIA
            # ------------------------------------------------

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


            clase_predicha = (
                prediccion.item()
            )

            confianza = (
                confianza_tensor.item()
            )


            suma_confianza += (
                confianza
            )


            # ------------------------------------------------
            # EXACTITUD CRUDA
            # ------------------------------------------------

            if (
                clase_predicha
                == clase_esperada
            ):

                frames_correctos += 1


            # ------------------------------------------------
            # FILTRO
            # ------------------------------------------------

            comando = filtro.update(
                clase_predicha,
                confianza
            )


            # ------------------------------------------------
            # COMANDO EMITIDO
            # ------------------------------------------------

            if comando is not None:

                comandos_emitidos += 1

                comandos_obtenidos.append(
                    comando
                )


                if (
                    comando
                    == clase_esperada
                ):

                    comandos_correctos += 1

                else:

                    falsos_comandos += 1


        else:

            filtro.reset()


        # ====================================================
        # GUARDAR FRAME
        # ====================================================

        predicciones_frames.append(
            {
                "ensayo": numero_ensayo,
                "condicion": condicion,
                "clase_esperada": clase_esperada,
                "mano_detectada":
                    int(
                        imagen_landmarks
                        is not None
                    ),
                "clase_predicha": clase_predicha,
                "confianza": confianza * 100.0,
                "comando_emitido":
                    (
                        comando
                        if comando is not None
                        else -1
                    )
            }
        )


        # ====================================================
        # VISUALIZACION
        # ====================================================

        restante = (
            DURACION_ENSAYO
            - (
                time.perf_counter()
                - tiempo_inicio
            )
        )


        cv2.putText(
            frame,
            f"Condicion: {condicion}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Esperada: {clase_esperada}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        if clase_predicha >= 0:

            cv2.putText(
                frame,
                f"Predicha: {clase_predicha}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Confianza: {confianza * 100:.1f}%",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )


        else:

            cv2.putText(
                frame,
                "MANO NO DETECTADA",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


        cv2.putText(
            frame,
            f"Tiempo: {max(restante, 0):.1f} s",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Falsos comandos: {falsos_comandos}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )


        cv2.imshow(
            "E6-B - Robustez",
            frame
        )


        cv2.waitKey(
            1
        )


    # --------------------------------------------------------
    # CALCULOS DEL ENSAYO
    # --------------------------------------------------------

    if frames_totales > 0:

        tasa_deteccion = (
            frames_detectados
            / frames_totales
            * 100.0
        )

    else:

        tasa_deteccion = 0.0


    if frames_detectados > 0:

        accuracy_raw = (
            frames_correctos
            / frames_detectados
            * 100.0
        )

        confianza_media = (
            suma_confianza
            / frames_detectados
            * 100.0
        )

    else:

        accuracy_raw = 0.0

        confianza_media = 0.0


    # --------------------------------------------------------
    # EXITO DEL COMANDO
    #
    # Para clase 0 no esperamos comando de accion.
    # El exito consiste en NO producir 1-4.
    # --------------------------------------------------------

    if clase_esperada == 0:

        exito_comando = int(
            falsos_comandos == 0
        )

    else:

        exito_comando = int(
            comandos_correctos > 0
        )


    resultado = {

        "ensayo": numero_ensayo,

        "condicion": condicion,

        "clase_esperada": clase_esperada,

        "frames_totales": frames_totales,

        "frames_detectados": frames_detectados,

        "tasa_deteccion_pct":
            tasa_deteccion,

        "frames_correctos":
            frames_correctos,

        "accuracy_raw_pct":
            accuracy_raw,

        "confianza_media_pct":
            confianza_media,

        "comandos_emitidos":
            comandos_emitidos,

        "comandos_correctos":
            comandos_correctos,

        "falsos_comandos":
            falsos_comandos,

        "exito_comando":
            exito_comando,

        "comandos_obtenidos":
            str(
                comandos_obtenidos
            )
    }


    print()
    print(
        "RESULTADO:"
    )

    print(
        f"Tasa deteccion: "
        f"{tasa_deteccion:.2f}%"
    )

    print(
        f"Accuracy raw: "
        f"{accuracy_raw:.2f}%"
    )

    print(
        f"Confianza media: "
        f"{confianza_media:.2f}%"
    )

    print(
        f"Comandos emitidos: "
        f"{comandos_emitidos}"
    )

    print(
        f"Comandos correctos: "
        f"{comandos_correctos}"
    )

    print(
        f"Falsos comandos: "
        f"{falsos_comandos}"
    )

    print(
        f"Exito del ensayo: "
        f"{exito_comando}"
    )


    return resultado


# ============================================================
# EJECUTAR TODOS LOS ENSAYOS
# ============================================================

numero_ensayo = 0


try:

    for condicion in CONDICIONES:

        print()
        print()
        print("##########################################")
        print(
            f"CONDICION: {condicion.upper()}"
        )
        print("##########################################")

        print(
            descripciones[
                condicion
            ]
        )


        for clase in CLASES:

            numero_ensayo += 1


            resultado = ejecutar_ensayo(
                condicion,
                clase,
                numero_ensayo
            )


            resultados_ensayos.append(
                resultado
            )


finally:

    camara.release()

    detector.close()

    cv2.destroyAllWindows()


# ============================================================
# GUARDAR ENSAYOS CSV
# ============================================================

ruta_ensayos = os.path.join(
    RESULTS_PATH,
    "robustez_ensayos.csv"
)


with open(
    ruta_ensayos,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    campos = [
        "ensayo",
        "condicion",
        "clase_esperada",
        "frames_totales",
        "frames_detectados",
        "tasa_deteccion_pct",
        "frames_correctos",
        "accuracy_raw_pct",
        "confianza_media_pct",
        "comandos_emitidos",
        "comandos_correctos",
        "falsos_comandos",
        "exito_comando",
        "comandos_obtenidos"
    ]


    writer = csv.DictWriter(
        archivo,
        fieldnames=campos
    )

    writer.writeheader()

    writer.writerows(
        resultados_ensayos
    )


# ============================================================
# GUARDAR FRAMES CSV
# ============================================================

ruta_frames = os.path.join(
    RESULTS_PATH,
    "robustez_frames.csv"
)


with open(
    ruta_frames,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    campos = [
        "ensayo",
        "condicion",
        "clase_esperada",
        "mano_detectada",
        "clase_predicha",
        "confianza",
        "comando_emitido"
    ]


    writer = csv.DictWriter(
        archivo,
        fieldnames=campos
    )

    writer.writeheader()

    writer.writerows(
        predicciones_frames
    )


# ============================================================
# RESUMEN POR CONDICION
# ============================================================

resumen_condiciones = []


for condicion in CONDICIONES:

    datos = [
        x
        for x in resultados_ensayos
        if x["condicion"] == condicion
    ]


    deteccion_media = np.mean(
        [
            x["tasa_deteccion_pct"]
            for x in datos
        ]
    )


    accuracy_media = np.mean(
        [
            x["accuracy_raw_pct"]
            for x in datos
        ]
    )


    confianza_media = np.mean(
        [
            x["confianza_media_pct"]
            for x in datos
        ]
    )


    total_comandos = sum(
        x["comandos_emitidos"]
        for x in datos
    )


    total_correctos = sum(
        x["comandos_correctos"]
        for x in datos
    )


    total_falsos = sum(
        x["falsos_comandos"]
        for x in datos
    )


    ensayos_exitosos = sum(
        x["exito_comando"]
        for x in datos
    )


    if total_comandos > 0:

        tasa_falsos = (
            total_falsos
            / total_comandos
            * 100.0
        )

    else:

        tasa_falsos = 0.0


    resumen_condiciones.append(
        {
            "condicion": condicion,

            "deteccion_media_pct":
                deteccion_media,

            "accuracy_media_pct":
                accuracy_media,

            "confianza_media_pct":
                confianza_media,

            "comandos_emitidos":
                total_comandos,

            "comandos_correctos":
                total_correctos,

            "falsos_comandos":
                total_falsos,

            "tasa_falsos_pct":
                tasa_falsos,

            "ensayos_exitosos":
                ensayos_exitosos,

            "ensayos_totales":
                len(datos)
        }
    )


# ============================================================
# CSV RESUMEN
# ============================================================

ruta_resumen_csv = os.path.join(
    RESULTS_PATH,
    "robustez_resumen_condiciones.csv"
)


with open(
    ruta_resumen_csv,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    campos = [
        "condicion",
        "deteccion_media_pct",
        "accuracy_media_pct",
        "confianza_media_pct",
        "comandos_emitidos",
        "comandos_correctos",
        "falsos_comandos",
        "tasa_falsos_pct",
        "ensayos_exitosos",
        "ensayos_totales"
    ]


    writer = csv.DictWriter(
        archivo,
        fieldnames=campos
    )

    writer.writeheader()

    writer.writerows(
        resumen_condiciones
    )


# ============================================================
# MOSTRAR RESUMEN
# ============================================================

print()
print()
print("==========================================")
print("E6-B - RESUMEN DE ROBUSTEZ")
print("==========================================")

print()

print(
    f"{'Condicion':<15}"
    f"{'Detec.%':<12}"
    f"{'Acc.%':<12}"
    f"{'Conf.%':<12}"
    f"{'Falsos':<10}"
    f"{'Exitos':<10}"
)


for x in resumen_condiciones:

    exitos_texto = (
        f"{x['ensayos_exitosos']}"
        f"/"
        f"{x['ensayos_totales']}"
    )


    print(
        f"{x['condicion']:<15}"
        f"{x['deteccion_media_pct']:<12.2f}"
        f"{x['accuracy_media_pct']:<12.2f}"
        f"{x['confianza_media_pct']:<12.2f}"
        f"{x['falsos_comandos']:<10}"
        f"{exitos_texto:<10}"
    )


# ============================================================
# GRAFICA ACCURACY Y DETECCION
# ============================================================

nombres = [
    x["condicion"]
    for x in resumen_condiciones
]


accuracy = [
    x["accuracy_media_pct"]
    for x in resumen_condiciones
]


deteccion = [
    x["deteccion_media_pct"]
    for x in resumen_condiciones
]


xpos = np.arange(
    len(nombres)
)

ancho = 0.35


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.bar(
    xpos - ancho / 2,
    accuracy,
    ancho,
    label="Accuracy"
)


ax.bar(
    xpos + ancho / 2,
    deteccion,
    ancho,
    label="Deteccion de mano"
)


ax.set_ylabel(
    "Porcentaje [%]"
)

ax.set_title(
    "E6-B - Robustez por condicion"
)

ax.set_xticks(
    xpos
)

ax.set_xticklabels(
    nombres
)

ax.set_ylim(
    0,
    105
)

ax.legend()


plt.tight_layout()


ruta_grafica = os.path.join(
    RESULTS_PATH,
    "robustez_por_condicion.png"
)


plt.savefig(
    ruta_grafica,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# GRAFICA FALSOS COMANDOS
# ============================================================

falsos = [
    x["falsos_comandos"]
    for x in resumen_condiciones
]


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.bar(
    nombres,
    falsos
)


ax.set_ylabel(
    "Numero de falsos comandos"
)

ax.set_title(
    "E6-B - Falsos comandos por condicion"
)


plt.tight_layout()


ruta_grafica_falsos = os.path.join(
    RESULTS_PATH,
    "falsos_comandos.png"
)


plt.savefig(
    ruta_grafica_falsos,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# TXT RESUMEN
# ============================================================

ruta_txt = os.path.join(
    RESULTS_PATH,
    "resumen_robustez.txt"
)


with open(
    ruta_txt,
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(
        "E6-B - ROBUSTEZ Y FALSOS COMANDOS\n"
    )

    archivo.write(
        "=================================\n\n"
    )


    for x in resumen_condiciones:

        archivo.write(
            f"Condicion: {x['condicion']}\n"
        )

        archivo.write(
            f"Deteccion media: "
            f"{x['deteccion_media_pct']:.2f}%\n"
        )

        archivo.write(
            f"Accuracy media: "
            f"{x['accuracy_media_pct']:.2f}%\n"
        )

        archivo.write(
            f"Confianza media: "
            f"{x['confianza_media_pct']:.2f}%\n"
        )

        archivo.write(
            f"Falsos comandos: "
            f"{x['falsos_comandos']}\n"
        )

        archivo.write(
            f"Ensayos exitosos: "
            f"{x['ensayos_exitosos']}"
            f"/"
            f"{x['ensayos_totales']}\n\n"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("==========================================")
print("ARCHIVOS GENERADOS")
print("==========================================")

print(
    ruta_ensayos
)

print(
    ruta_frames
)

print(
    ruta_resumen_csv
)

print(
    ruta_grafica
)

print(
    ruta_grafica_falsos
)

print(
    ruta_txt
)

print()
print("==========================================")
print("E6-B COMPLETADO")
print("==========================================")