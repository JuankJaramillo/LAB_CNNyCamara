from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print("==========================================")
print("PRUEBA DE CONEXION CON COPPELIASIM")
print("==========================================")


try:

    client = RemoteAPIClient()

    sim = client.require("sim")

    print("\nConexion establecida correctamente.")

    estado = sim.getSimulationState()

    print(
        f"Estado actual de simulacion: {estado}"
    )

    print("\nIniciando simulacion por 3 segundos...")


    # ========================================================
    # MODO SINCRONIZADO
    # ========================================================

    sim.setStepping(True)


    # ========================================================
    # INICIAR
    # ========================================================

    sim.startSimulation()


    while sim.getSimulationTime() < 3.0:

        tiempo = sim.getSimulationTime()

        print(
            f"Tiempo simulacion: {tiempo:.2f} s"
        )

        sim.step()


    # ========================================================
    # DETENER
    # ========================================================

    sim.stopSimulation()


    print("\nSimulacion detenida.")

    print(
        "Conexion Python <-> CoppeliaSim funcionando."
    )


except Exception as error:

    print("\nERROR DE CONEXION:")

    print(error)

    print(
        "\nVerifica que CoppeliaSim este abierto."
    )