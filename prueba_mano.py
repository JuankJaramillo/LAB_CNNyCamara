import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la camara.")
    exit()

print("Prueba de deteccion de mano.")
print("Presiona Q para salir.")


while True:

    ret, frame = camara.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    resultado = hands.process(rgb)

    if resultado.multi_hand_landmarks:

        for landmarks in resultado.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            alto, ancho, _ = frame.shape

            xs = []
            ys = []

            for punto in landmarks.landmark:

                xs.append(
                    int(punto.x * ancho)
                )

                ys.append(
                    int(punto.y * alto)
                )

            x1 = max(
                min(xs) - 50,
                0
            )

            y1 = max(
                min(ys) - 50,
                0
            )

            x2 = min(
                max(xs) + 50,
                ancho
            )

            y2 = min(
                max(ys) + 50,
                alto
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "MANO DETECTADA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
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

    cv2.imshow(
        "Prueba deteccion de mano",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camara.release()
hands.close()
cv2.destroyAllWindows()