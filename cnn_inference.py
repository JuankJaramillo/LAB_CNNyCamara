import cv2
import time
import torch

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


# ============================================================
# CONFIGURACION
# ============================================================

MODEL_PATH = (
    "models_v3/best_cnn_v3.pth"
)

CLASES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CARGAR CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# ============================================================
# CREAR MODELO
# ============================================================

modelo = FingerCNNV3(
    num_classes=5
)

modelo.load_state_dict(
    checkpoint["model_state_dict"]
)

modelo = modelo.to(
    device
)

modelo.eval()


# ============================================================
# INFORMACION DEL MODELO
# ============================================================

print(
    "=========================================="
)

print(
    "CNN V3 + LANDMARKS + FILTRO"
)

print(
    "=========================================="
)

print(
    f"\nDispositivo: {device}"
)

print(
    f"Modelo: {MODEL_PATH}"
)

print(
    f"Epoch seleccionada: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation Loss: "
    f"{checkpoint['val_loss']:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)

print()

print(
    "CLASES:"
)

print(
    "0 -> Sin comando"
)

print(
    "1 -> Joint 1"
)

print(
    "2 -> Joint 2"
)

print(
    "3 -> Joint 3"
)

print(
    "4 -> Gripper"
)

print()

print(
    "Presiona Q para salir."
)


# ============================================================
# FILTRO DE COMANDOS
# ============================================================

filtro = CommandFilter(
    window_size=7,
    min_votes=5,
    confidence_threshold=0.80,
    cooldown=0.8
)


# ============================================================
# DETECTOR MEDIAPIPE
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

    print(
        "ERROR: No se pudo abrir la camara."
    )

    detector.close()

    exit()


# ============================================================
# VARIABLES
# ============================================================

clase = "-"

confianza = 0.0

comando = None

ultimo_comando = "-"

tiempo_mostrar_comando = 0.0

DURACION_MENSAJE = 0.7


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:

    inicio = time.time()

    ret, frame = camara.read()


    if not ret:

        print(
            "ERROR: No se pudo leer la camara."
        )

        break


    # ========================================================
    # IMAGEN TIPO ESPEJO
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # GENERAR IMAGEN NORMALIZADA DE LANDMARKS
    # ========================================================

    imagen_landmarks, bbox = (
        crear_imagen_landmarks(
            frame,
            detector,
            size=128
        )
    )


    # Por defecto no hay nuevo comando
    comando = None


    # ========================================================
    # SI HAY MANO
    # ========================================================

    if imagen_landmarks is not None:

        # ====================================================
        # BOUNDING BOX
        # ====================================================

        x1, y1, x2, y2 = bbox


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ====================================================
        # CONVERTIR BGR -> RGB
        # ====================================================

        imagen_rgb = cv2.cvtColor(
            imagen_landmarks,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # NUMPY -> PIL
        # ====================================================

        imagen_pil = Image.fromarray(
            imagen_rgb
        )


        # ====================================================
        # PREPROCESAMIENTO
        # ====================================================

        tensor = eval_transform_v3(
            imagen_pil
        )


        tensor = tensor.unsqueeze(
            0
        )


        tensor = tensor.to(
            device
        )


        # ====================================================
        # INFERENCIA CNN
        # ====================================================

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


        # ====================================================
        # RESULTADO CNN
        # ====================================================

        clase_int = (
            prediccion.item()
        )


        clase = CLASES[
            clase_int
        ]


        confianza = (
            confianza_tensor.item()
            * 100.0
        )


        # ====================================================
        # FILTRO TEMPORAL
        # ====================================================

        comando = filtro.update(
            predicted_class=clase_int,
            confidence=(
                confianza / 100.0
            )
        )


        # ====================================================
        # SI SE GENERO UN NUEVO COMANDO
        # ====================================================

        if comando is not None:

            ultimo_comando = str(
                comando
            )

            tiempo_mostrar_comando = (
                time.time()
            )


            print(
                "=========================================="
            )

            print(
                f"COMANDO ESTABLE: {comando}"
            )

            print(
                f"Confianza CNN: {confianza:.1f}%"
            )

            if comando == 1:

                print(
                    "Accion futura: Joint 1"
                )

            elif comando == 2:

                print(
                    "Accion futura: Joint 2"
                )

            elif comando == 3:

                print(
                    "Accion futura: Joint 3"
                )

            elif comando == 4:

                print(
                    "Accion futura: Gripper"
                )

            print(
                "=========================================="
            )


        # ====================================================
        # MOSTRAR ENTRADA NORMALIZADA
        # ====================================================

        entrada_grande = cv2.resize(
            imagen_landmarks,
            (384, 384),
            interpolation=cv2.INTER_NEAREST
        )


        cv2.imshow(
            "Entrada normalizada CNN",
            entrada_grande
        )


    else:

        # ====================================================
        # SIN MANO
        # ====================================================

        clase = "-"

        confianza = 0.0

        filtro.reset()


    # ========================================================
    # FPS
    # ========================================================

    tiempo_proceso = (
        time.time()
        - inicio
    )


    if tiempo_proceso > 0:

        fps = (
            1.0
            / tiempo_proceso
        )

    else:

        fps = 0.0


    # ========================================================
    # MOSTRAR PREDICCION
    # ========================================================

    if imagen_landmarks is not None:

        cv2.putText(
            frame,
            f"Prediccion: {clase} dedos",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Confianza: {confianza:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # CLASE ESTABLE ACTUAL
        # ====================================================

        clase_estable = (
            filtro.get_stable_class()
        )


        if clase_estable is not None:

            cv2.putText(
                frame,
                f"Clase estable: {clase_estable}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


    else:

        cv2.putText(
            frame,
            "MANO NO DETECTADA",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # MOSTRAR COMANDO ESTABLE TEMPORALMENTE
    # ========================================================

    if (
        time.time()
        - tiempo_mostrar_comando
        < DURACION_MENSAJE
    ):

        cv2.putText(
            frame,
            f"COMANDO ESTABLE: {ultimo_comando}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2
        )


    # ========================================================
    # MOSTRAR FPS
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # INSTRUCCIONES
    # ========================================================

    cv2.putText(
        frame,
        "0: reposo | 1:J1 | 2:J2 | 3:J3 | 4:Gripper",
        (
            20,
            frame.shape[0] - 45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q: salir",
        (
            20,
            frame.shape[0] - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MOSTRAR CAMARA
    # ========================================================

    cv2.imshow(
        "CNN V3 - Reconocimiento y filtro",
        frame
    )


    # ========================================================
    # TECLADO
    # ========================================================

    tecla = (
        cv2.waitKey(1)
        & 0xFF
    )


    if tecla == ord("q"):

        break


# ============================================================
# FINALIZAR
# ============================================================

camara.release()

detector.close()

cv2.destroyAllWindows()


print()

print(
    "Inferencia finalizada."
)