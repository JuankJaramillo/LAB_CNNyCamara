import math
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# CONFIGURACION
# ============================================================

ANGULO_PRUEBA = 30.0
PASOS = 40
RETARDO = 0.03


# ============================================================
# MOVIMIENTO SUAVE
# ============================================================

def mover_suave(
    sim,
    joint,
    inicio,
    destino,
    pasos=40,
    retardo=0.03
):

    for i in range(1, pasos + 1):

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
print("PRUEBA VISUAL DE JOINTS UR5")
print("==========================================")

client = RemoteAPIClient()

sim = client.require("sim")


# ============================================================
# OBTENER UR5 Y JOINTS
# ============================================================

ur5 = sim.getObject("/UR5")

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
print(f"J1 = handle {J1}")
print(f"J2 = handle {J2}")
print(f"J3 = handle {J3}")


# ============================================================
# POSICIONES INICIALES
# ============================================================

q1_0 = sim.getJointPosition(J1)
q2_0 = sim.getJointPosition(J2)
q3_0 = sim.getJointPosition(J3)


delta = math.radians(
    ANGULO_PRUEBA
)


# ============================================================
# J1
# ============================================================

print()
print("==========================================")
print("PRUEBA J1 - BASE")
print("==========================================")

print(
    f"Moviendo J1 a +{ANGULO_PRUEBA:.0f} grados..."
)

mover_suave(
    sim,
    J1,
    q1_0,
    q1_0 + delta,
    PASOS,
    RETARDO
)

q_actual = sim.getJointPosition(J1)

print(
    f"Posicion leida J1: "
    f"{math.degrees(q_actual):.2f} grados"
)

input(
    "\nMira CoppeliaSim. "
    "Presiona ENTER para regresar J1..."
)

mover_suave(
    sim,
    J1,
    q1_0 + delta,
    q1_0,
    PASOS,
    RETARDO
)


# ============================================================
# J2
# ============================================================

print()
print("==========================================")
print("PRUEBA J2 - HOMBRO")
print("==========================================")

print(
    f"Moviendo J2 a +{ANGULO_PRUEBA:.0f} grados..."
)

mover_suave(
    sim,
    J2,
    q2_0,
    q2_0 + delta,
    PASOS,
    RETARDO
)

q_actual = sim.getJointPosition(J2)

print(
    f"Posicion leida J2: "
    f"{math.degrees(q_actual):.2f} grados"
)

input(
    "\nMira CoppeliaSim. "
    "Presiona ENTER para regresar J2..."
)

mover_suave(
    sim,
    J2,
    q2_0 + delta,
    q2_0,
    PASOS,
    RETARDO
)


# ============================================================
# J3
# ============================================================

print()
print("==========================================")
print("PRUEBA J3 - CODO")
print("==========================================")

print(
    f"Moviendo J3 a +{ANGULO_PRUEBA:.0f} grados..."
)

mover_suave(
    sim,
    J3,
    q3_0,
    q3_0 + delta,
    PASOS,
    RETARDO
)

q_actual = sim.getJointPosition(J3)

print(
    f"Posicion leida J3: "
    f"{math.degrees(q_actual):.2f} grados"
)

input(
    "\nMira CoppeliaSim. "
    "Presiona ENTER para regresar J3..."
)

mover_suave(
    sim,
    J3,
    q3_0 + delta,
    q3_0,
    PASOS,
    RETARDO
)


# ============================================================
# RESTAURAR
# ============================================================

sim.setJointPosition(
    J1,
    q1_0
)

sim.setJointPosition(
    J2,
    q2_0
)

sim.setJointPosition(
    J3,
    q3_0
)


print()
print("==========================================")
print("PRUEBA FINALIZADA")
print("==========================================")

print(
    "J1, J2 y J3 restaurados."
)