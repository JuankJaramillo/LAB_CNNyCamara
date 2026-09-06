import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands


def crear_detector(
    static_image_mode=False
):
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def extraer_mano(
    imagen_bgr,
    detector,
    margen=0.35
):

    alto, ancho, _ = imagen_bgr.shape

    imagen_rgb = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2RGB
    )

    resultado = detector.process(
        imagen_rgb
    )

    if not resultado.multi_hand_landmarks:
        return None, None

    landmarks = (
        resultado.multi_hand_landmarks[0]
    )

    xs = []
    ys = []

    for punto in landmarks.landmark:

        xs.append(
            int(punto.x * ancho)
        )

        ys.append(
            int(punto.y * alto)
        )

    xmin = min(xs)
    xmax = max(xs)

    ymin = min(ys)
    ymax = max(ys)

    ancho_mano = xmax - xmin
    alto_mano = ymax - ymin

    # Queremos un recorte cuadrado
    lado = max(
        ancho_mano,
        alto_mano
    )

    # Margen adicional alrededor de la mano
    lado = int(
        lado * (1.0 + margen)
    )

    centro_x = (
        xmin + xmax
    ) // 2

    centro_y = (
        ymin + ymax
    ) // 2

    x1 = centro_x - lado // 2
    y1 = centro_y - lado // 2

    x2 = x1 + lado
    y2 = y1 + lado

    # ========================================================
    # CALCULAR PADDING SI EL RECORTE SALE DE LA IMAGEN
    # ========================================================

    pad_left = max(
        0,
        -x1
    )

    pad_top = max(
        0,
        -y1
    )

    pad_right = max(
        0,
        x2 - ancho
    )

    pad_bottom = max(
        0,
        y2 - alto
    )

    x1_real = max(
        0,
        x1
    )

    y1_real = max(
        0,
        y1
    )

    x2_real = min(
        ancho,
        x2
    )

    y2_real = min(
        alto,
        y2
    )

    recorte = imagen_bgr[
        y1_real:y2_real,
        x1_real:x2_real
    ].copy()

    # Completar con blanco si faltó espacio
    recorte = cv2.copyMakeBorder(
        recorte,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    bbox = (
        x1_real,
        y1_real,
        x2_real,
        y2_real
    )

    return recorte, bbox