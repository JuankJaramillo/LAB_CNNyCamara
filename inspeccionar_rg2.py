from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print("==========================================")
print("INSPECCION DEL GRIPPER RG2")
print("==========================================")

client = RemoteAPIClient()

sim = client.require("sim")


# ============================================================
# BUSCAR RG2
# ============================================================

rg2 = sim.getObject("/UR5/connection/RG2")


print()
print(
    f"RG2 handle: {rg2}"
)


# ============================================================
# OBTENER TODOS LOS OBJETOS DEL RG2
# ============================================================

objetos = sim.getObjectsInTree(
    rg2,
    sim.handle_all,
    0
)


print()
print(
    f"Cantidad de objetos internos: {len(objetos)}"
)


print()
print("==========================================")
print("OBJETOS INTERNOS")
print("==========================================")


for i, handle in enumerate(
    objetos,
    start=1
):

    try:

        alias = sim.getObjectAlias(
            handle,
            -1
        )

    except Exception:

        alias = "SIN_ALIAS"


    try:

        ruta = sim.getObjectAlias(
            handle,
            2
        )

    except Exception:

        ruta = "SIN_RUTA"


    try:

        tipo = sim.getObjectType(
            handle
        )

    except Exception:

        tipo = -1


    try:

        posicion = sim.getObjectPosition(
            handle,
            sim.handle_world
        )

        x = posicion[0]
        y = posicion[1]
        z = posicion[2]

        posicion_texto = (
            f"x={x:.3f}, "
            f"y={y:.3f}, "
            f"z={z:.3f}"
        )

    except Exception:

        posicion_texto = (
            "posicion no disponible"
        )


    print()
    print(
        f"[{i}]"
    )

    print(
        f"Alias: {alias}"
    )

    print(
        f"Tipo: {tipo}"
    )

    print(
        f"Ruta: {ruta}"
    )

    print(
        f"Posicion: {posicion_texto}"
    )


print()
print("==========================================")
print("INSPECCION FINALIZADA")
print("==========================================")