# Configuración y modelo del sistema

## 1. Descripción

Este documento describe la configuración utilizada para el entrenamiento, inferencia y control robótico del sistema de reconocimiento de gestos mediante CNN.

El clasificador final reconoce cinco clases:

```text
0 dedos
1 dedo
2 dedos
3 dedos
4 dedos
```

La clasificación se utiliza posteriormente para generar comandos dirigidos al manipulador UR5 y al gripper RG2 en CoppeliaSim.

---

# 2. Entorno de desarrollo

Configuración utilizada:

```text
Sistema operativo: Windows
Python: 3.11
Framework CNN: PyTorch
Procesamiento de imagen: OpenCV
Detección de mano: MediaPipe
Simulación robótica: CoppeliaSim Edu
Comunicación: ZeroMQ Remote API
```

Durante las pruebas realizadas el modelo fue ejecutado en:

```text
CPU
```

---

# 3. Representación de entrada

La imagen original de la webcam no es utilizada directamente como entrada final de la CNN.

El procedimiento utilizado es:

```text
Imagen webcam
      ↓
MediaPipe Hands
      ↓
Detección de 21 landmarks
      ↓
Cálculo del bounding box
      ↓
Normalización geométrica
      ↓
Representación sobre fondo uniforme
      ↓
Imagen 128 × 128
      ↓
CNN
```

La representación final contiene únicamente la geometría principal de la mano.

Esto permite disminuir la influencia de:

```text
Fondo
Iluminación
Escala
Distancia
Posición dentro de la imagen
```

---

# 4. Tamaño de entrada

La entrada utilizada por la CNN es:

```text
Batch × 3 × 128 × 128
```

Una muestra individual corresponde a:

```text
3 × 128 × 128
```

---

# 5. Normalización

Después de convertir la imagen a tensor se utiliza:

```python
Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
```

---

# 6. Data augmentation

Durante entrenamiento se aplican transformaciones moderadas:

```text
Resize: 128 × 128

RandomHorizontalFlip:
p = 0.5

RandomRotation:
±6 grados

RandomAffine:
Traslación máxima = 5 %
Escala = 0.95 a 1.05
```

El objetivo es introducir pequeñas variaciones geométricas sin modificar la clase representada.

Durante validación, test e inferencia no se utiliza data augmentation.

---

# 7. Arquitectura CNN

Nombre:

```text
FingerCNNV3
```

Arquitectura:

```text
Input
3 × 128 × 128
```

## Bloque convolucional 1

```text
Conv2D
3 → 16
Kernel = 3 × 3
Padding = 1

BatchNorm2D
ReLU
MaxPool2D 2 × 2
```

Salida aproximada:

```text
16 × 64 × 64
```

## Bloque convolucional 2

```text
Conv2D
16 → 32
Kernel = 3 × 3
Padding = 1

BatchNorm2D
ReLU
MaxPool2D
```

Salida:

```text
32 × 32 × 32
```

## Bloque convolucional 3

```text
Conv2D
32 → 64
Kernel = 3 × 3
Padding = 1

BatchNorm2D
ReLU
MaxPool2D
```

Salida:

```text
64 × 16 × 16
```

## Bloque convolucional 4

```text
Conv2D
64 → 128
Kernel = 3 × 3
Padding = 1

BatchNorm2D
ReLU
MaxPool2D
```

Salida:

```text
128 × 8 × 8
```

## Pooling adaptativo

```text
AdaptiveAvgPool2D
Salida = 2 × 2
```

Resultado:

```text
128 × 2 × 2
```

## Flatten

```text
128 × 2 × 2 = 512 características
```

## Clasificador

```text
Linear
512 → 128

ReLU

Dropout
p = 0.5

Linear
128 → 5
```

Salida:

```text
5 logits
```

correspondientes a las clases:

```text
[0, 1, 2, 3, 4]
```

---

# 8. Número de parámetros

El modelo contiene:

```text
164229 parámetros
```

Todos los parámetros son entrenables.

---

# 9. Configuración de entrenamiento

```text
Optimizer:
Adam

Learning rate:
0.0005

Loss:
CrossEntropyLoss

Batch size:
32

Máximo de épocas:
30

Early stopping:
patience = 7
```

---

# 10. Partición del dataset

La partición fue realizada por sesión, no mediante división aleatoria de imágenes.

```text
TRAIN
Sesiones 01–04
1904 imágenes

VALIDATION
Sesión 05
574 imágenes

TEST
Sesión 06
563 imágenes
```

Esta separación evita que imágenes capturadas consecutivamente dentro de una misma sesión aparezcan simultáneamente en entrenamiento y evaluación.

---

# 11. Selección del modelo

La selección se realizó utilizando exclusivamente el conjunto de validación.

El criterio empleado fue:

```text
Menor Validation Loss
```

El checkpoint seleccionado fue:

```text
Epoch = 18

Validation Loss = 0.0159

Validation Accuracy = 99.83 %
```

Archivo:

```text
models_v3/best_cnn_v3.pth
```

El conjunto de test no fue utilizado para seleccionar ni modificar el modelo.

---

# 12. Resultado en test

Después de seleccionar definitivamente el checkpoint se evaluó una única vez sobre el conjunto de test independiente.

```text
Número de muestras:
563

Accuracy:
99.64 %
```

Matriz de confusión:

```text
[[113   0   0   0   0]
 [  0 110   0   0   0]
 [  0   0 109   0   0]
 [  0   0   2 110   0]
 [  0   0   0   0 119]]
```

Resultados por clase:

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 0.9820 | 1.0000 | 0.9909 |
| 3 | 1.0000 | 0.9821 | 0.9910 |
| 4 | 1.0000 | 1.0000 | 1.0000 |

---

# 13. Configuración del filtro de comandos

Para evitar que una predicción incorrecta aislada produzca movimiento del robot se utiliza un filtro temporal.

Parámetros:

```text
window_size = 7

min_votes = 5

confidence_threshold = 0.80

cooldown = 0.8 s
```

Una acción solo puede ser generada cuando:

```text
1. La confianza supera 80 %.

2. Al menos 5 de las últimas 7 predicciones
   corresponden a la misma clase.

3. El tiempo de cooldown permite emitir
   un nuevo comando.
```

La clase 0 funciona como:

```text
Reposo
Rearme
```

y permite posteriormente volver a ejecutar el mismo gesto.

---

# 14. Configuración del robot

Robot utilizado:

```text
UR5
```

Gripper:

```text
RG2
```

Se utilizan los primeros tres grados de libertad:

```text
J1
J2
J3
```

Mapeo:

```text
Clase 1 → J1
Clase 2 → J2
Clase 3 → J3
Clase 4 → RG2
```

---

# 15. Incremento de articulaciones

Cada comando modifica aproximadamente:

```text
10 grados
```

La dirección depende de la ubicación horizontal de la mano:

```text
Izquierda de la imagen → -10 grados

Derecha de la imagen → +10 grados
```

---

# 16. Control de posición

Durante la simulación se utiliza:

```python
sim.setJointTargetPosition(
    joint,
    destino
)
```

El uso de posición objetivo permite que el controlador dinámico de CoppeliaSim lleve la articulación hasta el ángulo deseado.

---

# 17. Control del RG2

La apertura y cierre del gripper se controla mediante:

```text
signal.RG2_open
```

Valores:

```text
1 → abrir
0 → cerrar
```

La clase 4 ejecuta un comportamiento tipo toggle:

```text
Abierto → cerrar
Cerrado → abrir
```

---

# 18. Comunicación con CoppeliaSim

La comunicación se realiza mediante:

```text
coppeliasim-zmqremoteapi-client
```

Flujo:

```text
Python
  ↓
ZeroMQ Remote API
  ↓
CoppeliaSim
  ↓
UR5 + RG2
```

---

# 19. Configuración de seguridad

Para disminuir movimientos accidentales se utilizan:

```text
Umbral de confianza
Filtro temporal
Cooldown
Clase 0 como rearme
Límites articulares
Movimiento incremental
```

Estas condiciones permiten desacoplar las predicciones instantáneas de las órdenes efectivamente ejecutadas.

---

# 20. Latencia medida

Resultados obtenidos durante operación en CPU:

```text
Percepción total:

p50 = 25.874 ms
p95 = 28.013 ms
```

```text
Filtro:

p50 = 0.031 ms
p95 = 0.047 ms
```

```text
Comando gripper:

p50 = 8.340 ms
p95 = 9.955 ms
```

La ejecución de articulaciones presentó una mediana de:

```text
54.631 ms
```

Durante uno de los ensayos J2 presentó un timeout aislado que incrementó artificialmente el p95 de ejecución.

---

# 21. Robustez

Se realizaron pruebas en:

```text
Condición normal
Mayor distancia
Posición lateral
Baja iluminación
```

Resultados:

| Condición | Accuracy |
|---|---:|
| Normal | 100.00 % |
| Lejos | 90.43 % |
| Lateral | 99.55 % |
| Baja luz | 87.43 % |

Se produjo un único falso comando durante los ensayos de baja iluminación.

---

# 22. Modelo final

El modelo final utilizado para todos los resultados presentados corresponde a:

```text
FingerCNNV3
Epoch 18
```

Checkpoint:

```text
models_v3/best_cnn_v3.pth
```

No se realizaron modificaciones al modelo después de evaluar el conjunto de test independiente.
