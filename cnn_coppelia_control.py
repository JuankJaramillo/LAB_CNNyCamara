import cv2
import time
import torch

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

from robot_controller import (
    RobotController
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
# CARGAR CNN
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


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


print("==========================================")
print("CNN + COPPELIASIM")
print("==========================================")

print(
    f"\nEpoch seleccionada: "
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
# MEDIAPIPE
# ============================================================

detector = crear_detector_landmarks(
    static_image_mode=False
)


# ============================================================
# COPPELIASIM
# ============================================================

robot = RobotController()


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

    robot.cerrar()

    detector.close()

    exit()


# ============================================================
# VARIABLES VISUALES
# ============================================================

ultimo_comando_texto = "NINGUNO"

tiempo_comando = 0.0

DURACION_MENSAJE = 1.0


print()
print("==========================================")
print("CONTROL ACTIVO")
print("==========================================")

print()
print("0 dedos -> Reposo / rearme")
print("1 dedo  -> Joint 1")
print("2 dedos -> Joint 2")
print("3 dedos -> Joint 3")
print("4 dedos -> Gripper")

print()
print(
    "Mano izquierda de pantalla -> -10 grados"
)

print(
    "Mano derecha de pantalla -> +10 grados"
)

print()
print(
    "Presiona Q para finalizar."
)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:

    while True:

        inicio = time.time()


        ret, frame = camara.read()


        if not ret:

            break


        # Imagen espejo
        frame = cv2.flip(
            frame,
            1
        )


        alto_frame, ancho_frame, _ = (
            frame.shape
        )


        # ====================================================
        # LANDMARKS NORMALIZADOS
        # ====================================================

        imagen_landmarks, bbox = (
            crear_imagen_landmarks(
                frame,
                detector,
                size=128
            )
        )


        clase = "-"

        confianza = 0.0

        comando = None

        direccion_texto = "-"


        # ====================================================
        # SI HAY MANO
        # ====================================================

        if imagen_landmarks is not None:

            x1, y1, x2, y2 = bbox


            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # CENTRO DE LA MANO
            # =================================================

            centro_mano_x = (
                x1 + x2
            ) // 2


            centro_pantalla = (
                ancho_frame // 2
            )


            if centro_mano_x < centro_pantalla:

                direccion = -1

                direccion_texto = (
                    "IZQUIERDA (-)"
                )

            else:

                direccion = 1

                direccion_texto = (
                    "DERECHA (+)"
                )


            # Línea central
            cv2.line(
                frame,
                (centro_pantalla, 0),
                (
                    centro_pantalla,
                    alto_frame
                ),
                (255, 0, 0),
                1
            )


            # =================================================
            # PREPARAR ENTRADA CNN
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
            # INFERENCIA CNN
            # =================================================

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


            # =================================================
            # FILTRO
            # =================================================

            comando = filtro.update(
                predicted_class=clase_int,
                confidence=(
                    confianza / 100.0
                )
            )


            # =================================================
            # EJECUTAR COMANDO
            # =================================================

            if comando is not None:

                # ---------------------------------------------
                # JOINT 1
                # ---------------------------------------------

                if comando == 1:

                    robot.mover_joint(
                        1,
                        direccion
                    )

                    ultimo_comando_texto = (
                        f"J1 "
                        f"{direccion_texto}"
                    )


                # ---------------------------------------------
                # JOINT 2
                # ---------------------------------------------

                elif comando == 2:

                    robot.mover_joint(
                        2,
                        direccion
                    )

                    ultimo_comando_texto = (
                        f"J2 "
                        f"{direccion_texto}"
                    )


                # ---------------------------------------------
                # JOINT 3
                # ---------------------------------------------

                elif comando == 3:

                    robot.mover_joint(
                        3,
                        direccion
                    )

                    ultimo_comando_texto = (
                        f"J3 "
                        f"{direccion_texto}"
                    )


                # ---------------------------------------------
                # GRIPPER
                # ---------------------------------------------

                elif comando == 4:

                    robot.toggle_gripper()

                    ultimo_comando_texto = (
                        "GRIPPER"
                    )


                tiempo_comando = (
                    time.time()
                )


            # =================================================
            # MOSTRAR ENTRADA CNN
            # =================================================

            entrada_grande = cv2.resize(
                imagen_landmarks,
                (320, 320),
                interpolation=cv2.INTER_NEAREST
            )


            cv2.imshow(
                "Entrada CNN",
                entrada_grande
            )


        else:

            filtro.reset()


        # ====================================================
        # FPS
        # ====================================================

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


        # ====================================================
        # INTERFAZ
        # ====================================================

        if imagen_landmarks is not None:

            cv2.putText(
                frame,
                f"Prediccion: {clase} dedos",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Confianza: {confianza:.1f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Direccion: {direccion_texto}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            clase_estable = (
                filtro.get_stable_class()
            )


            if clase_estable is not None:

                cv2.putText(
                    frame,
                    f"Clase estable: {clase_estable}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
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


        # ====================================================
        # ULTIMO COMANDO
        # ====================================================

        if (
            time.time()
            - tiempo_comando
            < DURACION_MENSAJE
        ):

            cv2.putText(
                frame,
                f"COMANDO: {ultimo_comando_texto}",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )


        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "0:reposo | 1:J1 | 2:J2 | 3:J3 | 4:Gripper",
            (
                20,
                alto_frame - 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Q: salir",
            (
                20,
                alto_frame - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        # ====================================================
        # MOSTRAR
        # ====================================================

        cv2.imshow(
            "CNN + Control UR5",
            frame
        )


        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        if tecla == ord("q"):

            break


# ============================================================
# FINALIZACION SEGURA
# ============================================================

finally:

    camara.release()

    detector.close()

    cv2.destroyAllWindows()

    robot.cerrar()

    print()
    print(
        "Sistema finalizado."
    )