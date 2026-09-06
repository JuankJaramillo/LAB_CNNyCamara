from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print("==========================================")
print("MEDICION DE ESCENA COPPELIASIM")
print("==========================================")

client = RemoteAPIClient()
sim = client.require("sim")


# ============================================================
# OBJETOS DE LA ESCENA
# ============================================================

objetos = {
    "Cubo": "/obj_cubo",
    "Cilindro": "/obj_cilindro",
    "Esfera": "/obj_esfera"
}


print()
print("OBJETOS:")
print("------------------------------------------")


for nombre, ruta in objetos.items():

    handle = sim.getObject(ruta)

    posicion = sim.getObjectPosition(
        handle,
        sim.handle_world
    )

    print(
        f"{nombre}: "
        f"x={posicion[0]:.3f}, "
        f"y={posicion[1]:.3f}, "
        f"z={posicion[2]:.3f}"
    )


# ============================================================
# BUSCAR CONNECTION DEL UR5
# ============================================================

ur5 = sim.getObject("/UR5")

todos = sim.getObjectsInTree(
    ur5,
    sim.handle_all,
    0
)


connection = None
rg2 = None


for handle in todos:

    try:

        alias = sim.getObjectAlias(
            handle,
            -1
        )

        if alias == "connection":
            connection = handle

        if alias == "RG2":
            rg2 = handle

    except Exception:
        pass


# ============================================================
# POSICION CONNECTION
# ============================================================

print()
print("EXTREMO DEL ROBOT:")
print("------------------------------------------")


if connection is not None:

    posicion = sim.getObjectPosition(
        connection,
        sim.handle_world
    )

    print(
        f"Connection UR5: "
        f"x={posicion[0]:.3f}, "
        f"y={posicion[1]:.3f}, "
        f"z={posicion[2]:.3f}"
    )

else:

    print(
        "No se encontro el objeto connection."
    )


# ============================================================
# POSICION RG2
# ============================================================

if rg2 is not None:

    posicion = sim.getObjectPosition(
        rg2,
        sim.handle_world
    )

    print(
        f"RG2: "
        f"x={posicion[0]:.3f}, "
        f"y={posicion[1]:.3f}, "
        f"z={posicion[2]:.3f}"
    )

else:

    print(
        "No se encontro el objeto RG2."
    )


# ============================================================
# JOINTS
# ============================================================

joints = sim.getObjectsInTree(
    ur5,
    sim.sceneobject_joint,
    0
)


print()
print("JOINTS:")
print("------------------------------------------")


for i, joint in enumerate(
    joints[:3],
    start=1
):

    posicion = sim.getObjectPosition(
        joint,
        sim.handle_world
    )

    print(
        f"J{i}: "
        f"x={posicion[0]:.3f}, "
        f"y={posicion[1]:.3f}, "
        f"z={posicion[2]:.3f}"
    )


print()
print("==========================================")
print("MEDICION FINALIZADA")
print("==========================================")
