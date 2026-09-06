import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# CONEXION
# ============================================================

print("==========================================")
print("PRUEBA DEL GRIPPER RG2")
print("==========================================")

client = RemoteAPIClient()

sim = client.require("sim")


# ============================================================
# VERIFICAR ESTADO DE SIMULACION
# ============================================================

estado = sim.getSimulationState()

print(
    f"\nEstado inicial de simulacion: {estado}"
)


# ============================================================
# INICIAR SIMULACION
# ============================================================

if estado == sim.simulation_stopped:

    print(
        "\nIniciando simulacion..."
    )

    sim.startSimulation()

    time.sleep(1.0)


# ============================================================
# ABRIR GRIPPER
# ============================================================

print()
print("==========================================")
print("ABRIENDO GRIPPER")
print("==========================================")

sim.setIntProperty(
    sim.handle_scene,
    "signal.RG2_open",
    1
)

print(
    "Comando enviado: RG2_open = 1"
)

print(
    "El gripper deberia estar abriendose."
)

input(
    "\nMira CoppeliaSim. "
    "Presiona ENTER cuando confirmes que abrio..."
)


# ============================================================
# CERRAR GRIPPER
# ============================================================

print()
print("==========================================")
print("CERRANDO GRIPPER")
print("==========================================")

sim.setIntProperty(
    sim.handle_scene,
    "signal.RG2_open",
    0
)

print(
    "Comando enviado: RG2_open = 0"
)

print(
    "El gripper deberia estar cerrandose."
)

input(
    "\nMira CoppeliaSim. "
    "Presiona ENTER cuando confirmes que cerro..."
)


# ============================================================
# ABRIR OTRA VEZ
# ============================================================

print()
print("==========================================")
print("ABRIENDO NUEVAMENTE")
print("==========================================")

sim.setIntProperty(
    sim.handle_scene,
    "signal.RG2_open",
    1
)

print(
    "Comando enviado: RG2_open = 1"
)

input(
    "\nPresiona ENTER para finalizar la prueba..."
)


# ============================================================
# DETENER SIMULACION
# ============================================================

print(
    "\nDeteniendo simulacion..."
)

sim.stopSimulation()


print()
print("==========================================")
print("PRUEBA FINALIZADA")
print("==========================================")

print(
    "Secuencia ejecutada:"
)

print(
    "ABRIR -> CERRAR -> ABRIR"
)