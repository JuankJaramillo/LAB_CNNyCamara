import math
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# CONFIGURACION
# ============================================================

ANGULO_PRUEBA = 15.0
PASOS = 30
RETARDO = 0.02


# ============================================================
# FUNCION PARA MOVIMIENTO SUAVE
# ============================================================

def mover_suave(
    sim,
    joint,
    inicio,
    destino,
    pasos=30,
    retardo=0.02
):

    for i in range(
        1,
        pasos + 1
    ):

        t = i / pasos

        posicion = (
            inicio
            + (destino - inicio) * t
        )

        sim.setJointPosition(
            joint,
            posicion
        )

        time.sleep(
            retardo
        )


# ============================================================
# CONEXION
# ============================================================

print("==========================================")
print("PRUEBA DE MOVIMIENTO UR5")
print("==========================================")

client = RemoteAPIClient()

sim = client.require(
    "sim"
)


# ============================================================
# OBTENER MODELO
# ============================================================

ur5 = sim.getObject(
    "/UR5"
)


# ============================================================
# OBTENER JOINTS
# ============================================================

joints = sim.getObjectsInTree(
    ur5,
    sim.sceneobject_joint,
    0
)


if len(joints) < 6:

    raise RuntimeError(
        "No se encontraron los 6 joints del UR5."
    )


J1 = joints[0]
J2 = joints[1]
J3 = joints[2]


print()
print(f"J1 -> handle {J1}")
print(f"J2 -> handle {J2}")
print(f"J3 -> handle {J3}")


# ============================================================
# GUARDAR POSICIONES INICIALES
# ============================================================

q1_inicial = sim.getJointPosition(
    J1
)

q2_inicial = sim.getJointPosition(
    J2
)

q3_inicial = sim.getJointPosition(
    J3
)


print()
print("Posiciones iniciales:")

print(
    f"J1 = {math.degrees(q1_inicial):.2f} grados"
)

print(
    f"J2 = {math.degrees(q2_inicial):.2f} grados"
)

print(
    f"J3 = {math.degrees(q3_inicial):.2f} grados"
)


# ============================================================
# ANGULO DE PRUEBA
# ============================================================

delta = math.radians(
    ANGULO_PRUEBA
)


# ============================================================
# PRUEBA J1
# ============================================================

print()
print("==========================================")
print("PROBANDO J1")
print("==========================================")

print(
    f"Moviendo J1 +{ANGULO_PRUEBA:.0f} grados..."
)

mover_suave(
    sim,
    J1,
    q1_inicial,
    q1_inicial + delta,
    PASOS,
    RETARDO
)

time.sleep(
    1
)

print(
    "Regresando J1..."
)

mover_suave(
    sim,
    J1,
    q1_inicial + delta,
    q1_inicial,
    PASOS,
    RETARDO
)

time.sleep(
    1
)


# ============================================================
# PRUEBA J2
# ============================================================

print()
print("==========================================")
print("PROBANDO J2")
print("==========================================")

print(
    f"Moviendo J2 +{ANGULO_PRUEBA:.0f} grados..."
)

mover_suave(
    sim,
    J2,
    q2_inicial,
    q2_inicial + delta,
    PASOS,
    RETARDO
)

time.sleep(
    1
)

print(
    "Regresando J2..."
)

mover_suave(
    sim,
    J2,
    q2_inicial + delta,
    q2_inicial,
    PASOS,
    RETARDO
)

time.sleep(
    1
)


# ============================================================
# PRUEBA J3
# ============================================================

print()
print("==========================================")
print("PROBANDO J3")
print("==========================================")

print(
    f"Moviendo J3 +{ANGULO_PRUEBA:.0f} grados..."
)

mover_suave(
    sim,
    J3,
    q3_inicial,
    q3_inicial + delta,
    PASOS,
    RETARDO
)

time.sleep(
    1
)

print(
    "Regresando J3..."
)

mover_suave(
    sim,
    J3,
    q3_inicial + delta,
    q3_inicial,
    PASOS,
    RETARDO
)

time.sleep(
    1
)


# ============================================================
# ASEGURAR POSICION ORIGINAL
# ============================================================

sim.setJointPosition(
    J1,
    q1_inicial
)

sim.setJointPosition(
    J2,
    q2_inicial
)

sim.setJointPosition(
    J3,
    q3_inicial
)


# ============================================================
# FINAL
# ============================================================

print()
print("==========================================")
print("PRUEBA FINALIZADA")
print("==========================================")

print(
    "J1, J2 y J3 regresaron "
    "a su posicion inicial."
)