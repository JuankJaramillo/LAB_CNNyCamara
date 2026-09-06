from coppeliasim_zmqremoteapi_client import RemoteAPIClient


client = RemoteAPIClient()
sim = client.require("sim")


if sim.getSimulationState() != sim.simulation_stopped:
    raise RuntimeError(
        "Deten la simulacion antes de ejecutar."
    )


# ============================================================
# CUBO - 0 grados
# ============================================================

soporte_cubo = sim.getObject(
    "/soporte_cubo"
)

obj_cubo = sim.getObject(
    "/obj_cubo"
)

sim.setObjectPosition(
    soporte_cubo,
    sim.handle_world,
    [-0.377, 0.050, 0.941]
)

sim.setObjectPosition(
    obj_cubo,
    sim.handle_world,
    [-0.377, 0.050, 1.001]
)


# ============================================================
# CILINDRO - aproximadamente -30 grados
# ============================================================

soporte_cilindro = sim.getObject(
    "/soporte_cilindro"
)

obj_cilindro = sim.getObject(
    "/obj_cilindro"
)

sim.setObjectPosition(
    soporte_cilindro,
    sim.handle_world,
    [-0.326, 0.239, 0.941]
)

sim.setObjectPosition(
    obj_cilindro,
    sim.handle_world,
    [-0.326, 0.239, 1.001]
)


# ============================================================
# ESFERA - aproximadamente -60 grados
# ============================================================

soporte_esfera = sim.getObject(
    "/soporte_esfera"
)

obj_esfera = sim.getObject(
    "/obj_esfera"
)

sim.setObjectPosition(
    soporte_esfera,
    sim.handle_world,
    [-0.189, 0.377, 0.941]
)

sim.setObjectPosition(
    obj_esfera,
    sim.handle_world,
    [-0.189, 0.377, 0.994]
)


print("==========================================")
print("PIEZAS REUBICADAS")
print("==========================================")

print("Cubo      -> J1 aprox.   0 grados")
print("Cilindro  -> J1 aprox. -30 grados")
print("Esfera    -> J1 aprox. -60 grados")
print("Mesa      -> J1 aprox. +90 grados")