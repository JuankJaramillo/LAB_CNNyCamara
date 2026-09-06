import cv2

from landmark_input import (
    crear_detector_landmarks,
    crear_imagen_landmarks
)


detector = crear_detector_landmarks(
    static_image_mode=False
)


camara = cv2.VideoCapture(0)


if not camara.isOpened():

    print(
        "ERROR: No se pudo abrir la camara."
    )

    exit()


print("==========================================")
print("PRUEBA DE ENTRADA NORMALIZADA")
print("==========================================")

print(
    "Presiona Q para salir."
)


while True:

    ret, frame = camara.read()

    if not ret:

        break


    frame = cv2.flip(
        frame,
        1
    )


    imagen_landmarks, bbox = (
        crear_imagen_landmarks(
            frame,
            detector
        )
    )


    if imagen_landmarks is not None:

        x1, y1, x2, y2 = bbox


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


        imagen_grande = cv2.resize(
            imagen_landmarks,
            (384, 384),
            interpolation=cv2.INTER_NEAREST
        )


        cv2.imshow(
            "Entrada normalizada para CNN",
            imagen_grande
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
        "Camara",
        frame
    )


    tecla = (
        cv2.waitKey(1)
        & 0xFF
    )


    if tecla == ord("q"):

        break


camara.release()

detector.close()

cv2.destroyAllWindows()