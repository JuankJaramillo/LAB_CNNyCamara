import cv2

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la camara.")
    exit()

print("Camara detectada correctamente.")
print("Presiona Q para cerrar.")

while True:
    ret, frame = camara.read()

    if not ret:
        print("ERROR: No se pudo obtener la imagen.")
        break

    cv2.imshow("Prueba de camara - LAB CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()

print("Prueba finalizada.")