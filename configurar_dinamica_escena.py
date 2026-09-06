from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print("==========================================")
print("CONFIGURACION DINAMICA DE LA ESCENA")
print("==========================================")

client = RemoteAPIClient()
sim = client.require("sim")


# ============================================================
# VERIFICAR SIMULACION DETENIDA
# ============================================================

if sim.getSimulationState() != sim.simulation_stopped:

    raise RuntimeError(
        "Deten la simulacion antes de ejecutar este script."
    )


# ============================================================
# SOPORTES Y MESA
# Deben permanecer fijos, pero permitir colisiones.
# ============================================================

estructuras_fijas = [
    "/soporte_cubo",
    "/soporte_cilindro",
    "/soporte_esfera",
    "/mesa_destino",
]


print()
print("ESTRUCTURAS FIJAS:")
print("------------------------------------------")


for ruta in estructuras_fijas:

    handle = sim.getObject(ruta)

    sim.setBoolProperty(
        handle,
        "dynamic",
        False
    )

    sim.setBoolProperty(
        handle,
        "respondable",
        True
    )

    print(
        f"{ruta}: "
        f"dynamic=False, respondable=True"
    )


# ============================================================
# OBJETOS
# Deben poder caer, moverse y ser agarrados.
# ============================================================

objetos = [
    "/obj_cubo",
    "/obj_cilindro",
    "/obj_esfera",
]


print()
print("OBJETOS MANIPULABLES:")
print("------------------------------------------")


for ruta in objetos:

    handle = sim.getObject(ruta)

    sim.setBoolProperty(
        handle,
        "dynamic",
        True
    )

    sim.setBoolProperty(
        handle,
        "respondable",
        True
    )

    print(
        f"{ruta}: "
        f"dynamic=True, respondable=True"
    )


print()
print("==========================================")
print("CONFIGURACION COMPLETADA")
print("==========================================")