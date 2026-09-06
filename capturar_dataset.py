import cv2
import os
import time

# ============================================================
# CONFIGURACION DE LA SESION
# ============================================================

PERSONA = "persona_01"
SESION = "sesion_06"

CARPETA_DATASET = "dataset_raw"

CLASES = ["0", "1", "2", "3", "4"]

INTERVALO_CAPTURA = 0.25

# ============================================================
# CREAR CARPETAS
# ============================================================

for clase in CLASES:
    ruta = os.path.join(
        CARPETA_DATASET,
        PERSONA,
        SESION,
        clase
    )

    os.makedirs(ruta, exist_ok=True)

# ============================================================
# CONTADORES
# ============================================================

contadores = {}

for clase in CLASES:
    ruta = os.path.join(
        CARPETA_DATASET,
        PERSONA,
        SESION,
        clase
    )

    archivos = [
        f for f in os.listdir(ruta)
        if f.lower().endswith(".jpg")
    ]

    contadores[clase] = len(archivos)

# ============================================================
# CAMARA
# ============================================================

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la camara.")
    exit()

clase_actual = "0"
capturando = False
ultimo_guardado = 0

print("==========================================")
print("CAPTURA DE DATASET CNN")
print("==========================================")
print("Teclas 0-4 : seleccionar cantidad de dedos")
print("ESPACIO    : iniciar/detener captura")
print("Q          : salir")
print("==========================================")

while True:

    ret, frame = camara.read()

    if not ret:
        print("ERROR: No se pudo leer la camara.")
        break

    frame = cv2.flip(frame, 1)

    alto, ancho, _ = frame.shape

    # Region de interes
    x1 = int(ancho * 0.50)
    y1 = int(alto * 0.15)
    x2 = int(ancho * 0.95)
    y2 = int(alto * 0.90)

    roi = frame[y1:y2, x1:x2].copy()

    # Dibujar region de interes
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Captura automatica
    if capturando:

        tiempo_actual = time.time()

        if tiempo_actual - ultimo_guardado >= INTERVALO_CAPTURA:

            ruta = os.path.join(
                CARPETA_DATASET,
                PERSONA,
                SESION,
                clase_actual
            )

            nombre = (
                f"{PERSONA}_{SESION}_"
                f"clase_{clase_actual}_"
                f"{contadores[clase_actual]:04d}.jpg"
            )

            ruta_completa = os.path.join(ruta, nombre)

            cv2.imwrite(ruta_completa, roi)

            contadores[clase_actual] += 1

            ultimo_guardado = tiempo_actual

    # Informacion en pantalla
    cv2.putText(
        frame,
        f"Clase: {clase_actual} dedos",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Imagenes: {contadores[clase_actual]}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    estado = "CAPTURANDO" if capturando else "PAUSA"

    cv2.putText(
        frame,
        estado,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if capturando else (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "0-4: clase | ESPACIO: capturar | Q: salir",
        (20, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("Captura Dataset CNN", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break

    elif tecla in [
        ord("0"),
        ord("1"),
        ord("2"),
        ord("3"),
        ord("4")
    ]:
        clase_actual = chr(tecla)
        capturando = False

        print(
            f"Clase seleccionada: {clase_actual} dedos"
        )

    elif tecla == 32:
        capturando = not capturando

        if capturando:
            print(
                f"Capturando clase {clase_actual}..."
            )
        else:
            print("Captura pausada.")

camara.release()
cv2.destroyAllWindows()

print("\n==========================================")
print("RESUMEN DE CAPTURA")
print("==========================================")

for clase in CLASES:
    print(
        f"Clase {clase}: {contadores[clase]} imagenes"
    )

print("Programa finalizado.")