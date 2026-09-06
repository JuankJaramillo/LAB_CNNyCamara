import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print("==========================================")
print("IDENTIFICACION DE JOINTS DEL UR5")
print("==========================================")


try:

    # ========================================================
    # CONEXION
    # ========================================================

    client = RemoteAPIClient()

    sim = client.require("sim")


    # ========================================================
    # OBTENER MODELO UR5
    # ========================================================

    ur5 = sim.getObject("/UR5")


    print(
        f"\nHandle del UR5: {ur5}"
    )


    # ========================================================
    # OBTENER TODOS LOS JOINTS DENTRO DEL UR5
    # ========================================================

    joints = sim.getObjectsInTree(
        ur5,
        sim.sceneobject_joint,
        0
    )


    print(
        f"\nCantidad de joints encontrados: "
        f"{len(joints)}"
    )


    print("\n==========================================")
    print("JOINTS EN ORDEN DE LA CADENA")
    print("==========================================")


    # ========================================================
    # MOSTRAR INFORMACION
    # ========================================================

    for i, joint in enumerate(
        joints,
        start=1
    ):

        alias = sim.getObjectAlias(
            joint,
            -1
        )

        ruta = sim.getObjectAlias(
            joint,
            2
        )

        posicion_rad = sim.getJointPosition(
            joint
        )

        posicion_deg = math.degrees(
            posicion_rad
        )


        print()

        print(
            f"J{i}"
        )

        print(
            f"  Handle: {joint}"
        )

        print(
            f"  Alias: {alias}"
        )

        print(
            f"  Ruta: {ruta}"
        )

        print(
            f"  Posicion: "
            f"{posicion_rad:.4f} rad "
            f"({posicion_deg:.2f} grados)"
        )


    print()

    print("==========================================")
    print("IDENTIFICACION FINALIZADA")
    print("==========================================")


except Exception as error:

    print()

    print(
        "ERROR:"
    )

    print(
        error
    )