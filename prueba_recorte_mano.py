import cv2

from hand_crop import (
    crear_detector,
    extraer_mano
)


detector = crear_detector(
    static_image_mode=False
)

camara = cv2.VideoCapture(0)

if not camara.isOpened():

    print(
        "ERROR: No se pudo abrir la camara."
    )

    exit()


print("==========================================")
print("PRUEBA DE RECORTE DE MANO")
print("==========================================")
print("Presiona Q para salir.")


while True:

    ret, frame = camara.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    recorte, bbox = extraer_mano(
        frame,
        detector
    )


    if recorte is not None:

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

        # Este será aproximadamente
        # el tipo de imagen que verá la CNN
        recorte_mostrar = cv2.resize(
            recorte,
            (256, 256)
        )

        cv2.imshow(
            "Recorte para CNN",
            recorte_mostrar
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
        "Deteccion de mano",
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