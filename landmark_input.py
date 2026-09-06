import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands


def crear_detector_landmarks(
    static_image_mode=False
):

    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def crear_imagen_landmarks(
    imagen_bgr,
    detector,
    size=128,
    margen=0.12
):

    # ========================================================
    # BGR -> RGB
    # ========================================================

    imagen_rgb = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2RGB
    )

    resultado = detector.process(
        imagen_rgb
    )


    # ========================================================
    # VERIFICAR MANO
    # ========================================================

    if not resultado.multi_hand_landmarks:

        return None, None


    landmarks = (
        resultado.multi_hand_landmarks[0]
    )


    # ========================================================
    # EXTRAER COORDENADAS NORMALIZADAS
    # ========================================================

    puntos = np.array(
        [
            [p.x, p.y]
            for p in landmarks.landmark
        ],
        dtype=np.float32
    )


    xmin = puntos[:, 0].min()
    xmax = puntos[:, 0].max()

    ymin = puntos[:, 1].min()
    ymax = puntos[:, 1].max()


    ancho = xmax - xmin
    alto = ymax - ymin

    lado = max(
        ancho,
        alto
    )


    if lado <= 0:

        return None, None


    # ========================================================
    # CENTRAR Y NORMALIZAR ESCALA
    # ========================================================

    centro_x = (
        xmin + xmax
    ) / 2.0

    centro_y = (
        ymin + ymax
    ) / 2.0


    escala = (
        1.0 - 2.0 * margen
    ) / lado


    puntos_normalizados = np.zeros_like(
        puntos
    )


    puntos_normalizados[:, 0] = (
        (puntos[:, 0] - centro_x)
        * escala
        + 0.5
    )


    puntos_normalizados[:, 1] = (
        (puntos[:, 1] - centro_y)
        * escala
        + 0.5
    )


    # ========================================================
    # CREAR CANVAS BLANCO
    # ========================================================

    canvas = np.full(
        (
            size,
            size,
            3
        ),
        255,
        dtype=np.uint8
    )


    # ========================================================
    # PASAR PUNTOS A PIXELES
    # ========================================================

    puntos_px = []

    for x, y in puntos_normalizados:

        px = int(
            np.clip(
                x * (size - 1),
                0,
                size - 1
            )
        )

        py = int(
            np.clip(
                y * (size - 1),
                0,
                size - 1
            )
        )

        puntos_px.append(
            (px, py)
        )


    # ========================================================
    # DIBUJAR CONEXIONES
    # ========================================================

    for inicio, fin in mp_hands.HAND_CONNECTIONS:

        p1 = puntos_px[
            int(inicio)
        ]

        p2 = puntos_px[
            int(fin)
        ]

        cv2.line(
            canvas,
            p1,
            p2,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )


    # ========================================================
    # DIBUJAR LANDMARKS
    # ========================================================

    for punto in puntos_px:

        cv2.circle(
            canvas,
            punto,
            4,
            (0, 0, 0),
            -1,
            cv2.LINE_AA
        )


    # ========================================================
    # BOUNDING BOX EN IMAGEN ORIGINAL
    # ========================================================

    alto_img, ancho_img, _ = imagen_bgr.shape


    x1 = int(
        xmin * ancho_img
    )

    y1 = int(
        ymin * alto_img
    )

    x2 = int(
        xmax * ancho_img
    )

    y2 = int(
        ymax * alto_img
    )


    bbox = (
        x1,
        y1,
        x2,
        y2
    )


    return canvas, bbox