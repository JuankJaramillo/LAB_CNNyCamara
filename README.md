# Sistema de control robótico mediante CNN y visión artificial

## Laboratorio de Inteligencia Artificial

Este proyecto implementa un sistema de visión artificial capaz de reconocer en tiempo real gestos correspondientes a una cantidad de dedos entre 0 y 4 y utilizar dichas clases para controlar un manipulador robótico UR5 con gripper RG2 en CoppeliaSim.

El sistema integra:

- Captura de video mediante webcam.
- Detección de mano mediante MediaPipe.
- Normalización de 21 landmarks de la mano.
- Clasificación mediante una red neuronal convolucional CNN desarrollada en PyTorch.
- Filtro temporal para evitar comandos inestables.
- Comunicación entre Python y CoppeliaSim mediante ZeroMQ Remote API.
- Control de tres grados de libertad del UR5.
- Apertura y cierre del gripper RG2.
- Manipulación y traslado de objetos.
- Evaluación mediante test independiente.
- Pruebas de latencia, robustez y falsos comandos.

---

## 1. Funcionamiento general

El flujo del sistema es:

```text
Webcam
   ↓
MediaPipe
   ↓
21 landmarks de la mano
   ↓
Normalización a imagen 128x128
   ↓
CNN V3
   ↓
Clase 0, 1, 2, 3 o 4
   ↓
Filtro temporal de comandos
   ↓
Control del UR5 + RG2
   ↓
CoppeliaSim
```

La CNN es el clasificador final del sistema. MediaPipe se utiliza únicamente como etapa de detección y normalización geométrica de la mano.

---

## 2. Clases utilizadas

El sistema reconoce cinco clases:

| Clase | Gesto | Acción |
|---|---|---|
| 0 | 0 dedos | Reposo / rearme |
| 1 | 1 dedo | Control Joint 1 |
| 2 | 2 dedos | Control Joint 2 |
| 3 | 3 dedos | Control Joint 3 |
| 4 | 4 dedos | Abrir / cerrar gripper |

Para los joints J1, J2 y J3, la posición horizontal de la mano permite definir el sentido del movimiento:

```text
Mano a la izquierda → movimiento negativo
Mano a la derecha   → movimiento positivo
```

Cada comando modifica el objetivo angular en pasos de aproximadamente 10 grados.

---

## 3. Dataset

El dataset fue capturado mediante webcam en diferentes sesiones para evitar realizar una partición aleatoria de imágenes altamente correlacionadas.

El conjunto final contiene:

```text
Total: 3041 imágenes
```

Distribución utilizada:

```text
Train:
Sesiones 01, 02, 03 y 04
1904 imágenes

Validation:
Sesión 05
574 imágenes

Test:
Sesión 06
563 imágenes
```

El conjunto de test permaneció independiente durante todo el proceso de entrenamiento y selección del modelo.

---

## 4. Preprocesamiento

Inicialmente se trabajó directamente con regiones de la imagen de la mano. Sin embargo, se identificaron diferencias importantes de iluminación, escala y fondo entre sesiones.

Para mejorar la generalización se utilizó MediaPipe Hands para detectar 21 landmarks.

Los landmarks son normalizados respecto al tamaño y posición de la mano y posteriormente dibujados sobre una imagen uniforme de:

```text
128 x 128 píxeles
```

De esta forma, la CNN recibe principalmente información geométrica de la configuración de los dedos y se reduce la dependencia del fondo y la iluminación.

---

## 5. Arquitectura CNN final

El modelo utilizado es `FingerCNNV3`.

Arquitectura:

```text
Entrada: 3 x 128 x 128

Conv2D 3 → 16
BatchNorm
ReLU
MaxPool

Conv2D 16 → 32
BatchNorm
ReLU
MaxPool

Conv2D 32 → 64
BatchNorm
ReLU
MaxPool

Conv2D 64 → 128
BatchNorm
ReLU
MaxPool

AdaptiveAvgPool 2 x 2

Flatten

Linear 512 → 128
ReLU
Dropout 0.5

Linear 128 → 5
```

Número total de parámetros entrenables:

```text
164229
```

---

## 6. Entrenamiento

Configuración principal:

```text
Framework: PyTorch
Optimizer: Adam
Learning rate: 0.0005
Batch size: 32
Máximo de épocas: 30
Early stopping patience: 7
Función de pérdida: CrossEntropyLoss
```

El modelo fue seleccionado utilizando únicamente el conjunto de validación.

Checkpoint seleccionado:

```text
Época: 18
Validation Loss: 0.0159
Validation Accuracy: 99.83 %
```

Archivo:

```text
models_v3/best_cnn_v3.pth
```

---

## 7. Evaluación sobre test independiente

El modelo seleccionado fue evaluado posteriormente sobre 563 imágenes correspondientes a una sesión completamente independiente.

Resultado:

```text
Accuracy test: 99.64 %
```

Matriz de confusión:

```text
[[113   0   0   0   0]
 [  0 110   0   0   0]
 [  0   0 109   0   0]
 [  0   0   2 110   0]
 [  0   0   0   0 119]]
```

Únicamente se presentaron dos errores, ambos correspondientes a muestras reales de clase 3 clasificadas como clase 2.

---

## 8. Filtro de comandos

Para reducir comandos accidentales se implementó un filtro temporal.

Configuración:

```text
Ventana temporal: 7 predicciones
Votos mínimos: 5
Confianza mínima: 80 %
Cooldown: 0.8 s
```

Un comando solo es enviado al robot cuando existe suficiente consistencia temporal en las predicciones.

La clase 0 se utiliza como estado de reposo y rearme.

---

## 9. CoppeliaSim

Se utilizó:

```text
Robot: UR5
Gripper: RG2
```

La comunicación entre Python y CoppeliaSim se realiza mediante:

```text
ZeroMQ Remote API
```

Se controlan:

```text
Clase 1 → Joint 1
Clase 2 → Joint 2
Clase 3 → Joint 3
Clase 4 → RG2
```

Las articulaciones son comandadas mediante posiciones objetivo usando:

```python
sim.setJointTargetPosition()
```

El gripper se controla mediante la señal:

```text
signal.RG2_open
```

---

## 10. Manipulación de objetos

El entorno de CoppeliaSim contiene tres tipos de objetos:

```text
Cubo
Cilindro
Esfera
```

El sistema permite:

```text
1. Aproximar el manipulador.
2. Cerrar el gripper.
3. Sujetar el objeto.
4. Desplazarlo mediante los joints.
5. Transportarlo hasta una zona de destino.
6. Abrir el gripper.
7. Depositar el objeto.
```

---

## 11. Latencia

Se evaluó la latencia de las etapas del sistema.

Sobre 661 frames:

```text
Percepción total:
p50 = 25.874 ms
p95 = 28.013 ms

Filtro:
p50 = 0.031 ms
p95 = 0.047 ms

Gripper:
p50 = 8.340 ms
p95 = 9.955 ms
```

La ejecución de articulaciones presentó una mediana aproximada de:

```text
54.631 ms
```

Durante una prueba se registró un timeout aislado de J2, por lo que dicho evento se considera una condición anómala y no representa el comportamiento nominal del sistema.

---

## 12. Robustez

Se realizaron 20 ensayos en vivo bajo cuatro condiciones:

```text
Normal
Mayor distancia
Posición lateral
Baja iluminación
```

Resumen:

| Condición | Detección | Accuracy | Falsos comandos | Ensayos exitosos |
|---|---:|---:|---:|---:|
| Normal | 100.00 % | 100.00 % | 0 | 5/5 |
| Lejos | 99.77 % | 90.43 % | 0 | 4/5 |
| Lateral | 99.78 % | 99.55 % | 0 | 5/5 |
| Baja luz | 99.77 % | 87.43 % | 1 | 5/5 |

La condición más crítica fue la baja iluminación.

El filtro temporal evitó que la mayoría de errores aislados de clasificación se convirtieran en comandos enviados al robot.

---

## 13. Archivos principales

### `cnn_inference.py`

Realiza la captura de cámara, procesamiento de landmarks e inferencia mediante la CNN.

### `command_filter.py`

Implementa el filtro temporal y el umbral de confianza.

### `robot_adapter.py`

Interfaz entre las órdenes del sistema de visión y el robot simulado.

### `robot_controller.py`

Implementa la comunicación directa con CoppeliaSim y el control de J1, J2, J3 y RG2.

### `cnn_coppelia_control.py`

Integra visión, CNN, filtro y control del robot en tiempo real.

### `landmark_input.py`

Genera la representación normalizada de los 21 landmarks.

### `modelo_cnn_v3.py`

Define la arquitectura final de la CNN.

### `transforms_cnn_v3.py`

Contiene las transformaciones utilizadas para entrenamiento e inferencia.

---

## 14. Instalación

Se recomienda Python 3.11.

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Además se requiere:

```text
CoppeliaSim Edu
```

con soporte de ZeroMQ Remote API.

---

## 15. Ejecución

Primero abrir en CoppeliaSim la escena que contiene:

```text
UR5
RG2
Objetos
Zona de destino
```

Luego ejecutar:

```bash
python cnn_coppelia_control.py
```

La cámara se iniciará automáticamente.

Para salir:

```text
Q
```

---

## 16. Evaluación

Evaluación del test independiente:

```bash
python evaluar_test_final.py
```

Prueba de latencia:

```bash
python e6_medicion_latencia.py
```

Prueba de robustez:

```bash
python e6_robustez.py
```

---

## 17. Resultados principales

```text
Validation Accuracy: 99.83 %
Test Accuracy:       99.64 %

Test:
561 / 563 muestras correctamente clasificadas

Robustez normal:
100 % accuracy

Falsos comandos:
1 evento durante los ensayos de robustez
```

---

## 18. Conclusión

El sistema desarrollado permite reconocer gestos de mano mediante una CNN y utilizarlos para controlar un manipulador UR5 con gripper RG2 en CoppeliaSim.

La normalización mediante landmarks permitió reducir la dependencia del fondo, iluminación, posición y escala de la mano, obteniendo una exactitud del 99.64 % sobre un conjunto de prueba independiente.

La incorporación de un filtro temporal permitió reducir la ejecución de comandos incorrectos y mejorar la estabilidad del control robótico en tiempo real.
