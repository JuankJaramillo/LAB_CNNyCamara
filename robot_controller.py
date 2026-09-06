import math
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class RobotController:

    def __init__(self):

        print("==========================================")
        print("CONECTANDO CON COPPELIASIM")
        print("==========================================")

        # ====================================================
        # CONEXION
        # ====================================================

        self.client = RemoteAPIClient()

        self.sim = self.client.require(
            "sim"
        )


        # ====================================================
        # OBTENER UR5
        # ====================================================

        self.ur5 = self.sim.getObject(
            "/UR5"
        )

        print(
            f"\nUR5 -> handle {self.ur5}"
        )


        # ====================================================
        # DESACTIVAR SCRIPT DEL UR5
        # ====================================================

        try:

            self.ur5_script = self.sim.getObject(
                "/UR5/Script"
            )

            print(
                f"Script UR5 -> handle "
                f"{self.ur5_script}"
            )

            self.sim.setBoolProperty(
                self.ur5_script,
                "scriptDisabled",
                True
            )

            print(
                "Script interno del UR5: DESACTIVADO"
            )

        except Exception as error:

            print(
                "ADVERTENCIA: "
                "no se pudo desactivar el script del UR5."
            )

            print(error)


        # ====================================================
        # OBTENER JOINTS
        # ====================================================

        joints = self.sim.getObjectsInTree(
            self.ur5,
            self.sim.sceneobject_joint,
            0
        )


        if len(joints) < 6:

            raise RuntimeError(
                "No se encontraron los 6 joints del UR5."
            )


        self.J1 = joints[0]
        self.J2 = joints[1]
        self.J3 = joints[2]


        print()

        print(
            f"J1 -> handle {self.J1}"
        )

        print(
            f"J2 -> handle {self.J2}"
        )

        print(
            f"J3 -> handle {self.J3}"
        )


        # ====================================================
        # MOSTRAR MODO DE LOS JOINTS
        # ====================================================

        print()
        print("Modos de articulacion:")

        for numero, joint in [
            (1, self.J1),
            (2, self.J2),
            (3, self.J3)
        ]:

            modo = self.sim.getJointMode(
                joint
            )

            print(
                f"J{numero} -> modo {modo}"
            )


        # ====================================================
        # LIMITES DE TRABAJO
        # ====================================================

        self.limites = {

            1: (
                math.radians(-100),
                math.radians(100)
            ),

            2: (
                math.radians(-45),
                math.radians(45)
            ),

            3: (
                math.radians(-60),
                math.radians(60)
            )
        }


        # Cada gesto mueve 10 grados
        self.incremento = math.radians(
            10
        )


        # ====================================================
        # GRIPPER
        # ====================================================

        self.gripper_abierto = True


        # ====================================================
        # POSICIONES INICIALES
        # ====================================================

        self.q1_inicial = self.sim.getJointPosition(
            self.J1
        )

        self.q2_inicial = self.sim.getJointPosition(
            self.J2
        )

        self.q3_inicial = self.sim.getJointPosition(
            self.J3
        )


        print()
        print("Posiciones iniciales:")

        print(
            f"J1 = "
            f"{math.degrees(self.q1_inicial):.1f} grados"
        )

        print(
            f"J2 = "
            f"{math.degrees(self.q2_inicial):.1f} grados"
        )

        print(
            f"J3 = "
            f"{math.degrees(self.q3_inicial):.1f} grados"
        )


        # ====================================================
        # ESTABLECER TARGETS INICIALES
        # ====================================================

        self.sim.setJointTargetPosition(
            self.J1,
            self.q1_inicial
        )

        self.sim.setJointTargetPosition(
            self.J2,
            self.q2_inicial
        )

        self.sim.setJointTargetPosition(
            self.J3,
            self.q3_inicial
        )


        # ====================================================
        # INICIAR SIMULACION
        # ====================================================

        estado = self.sim.getSimulationState()


        if estado == self.sim.simulation_stopped:

            print()

            print(
                "Iniciando simulacion..."
            )

            self.sim.startSimulation()

            time.sleep(
                1.0
            )


        # ====================================================
        # GRIPPER ABIERTO INICIALMENTE
        # ====================================================

        self.sim.setIntProperty(
            self.sim.handle_scene,
            "signal.RG2_open",
            1
        )


        print()
        print(
            "Robot listo para control externo."
        )

        print(
            "Control de joints: TARGET POSITION"
        )

        print(
            "Script RG2 permanece activo."
        )


    # ========================================================
    # SELECCIONAR JOINT
    # ========================================================

    def _obtener_joint(
        self,
        numero_joint
    ):

        if numero_joint == 1:

            return self.J1

        elif numero_joint == 2:

            return self.J2

        elif numero_joint == 3:

            return self.J3

        return None


    # ========================================================
    # MOVER ARTICULACION
    # ========================================================

    def mover_joint(
        self,
        numero_joint,
        direccion
    ):

        joint = self._obtener_joint(
            numero_joint
        )


        if joint is None:

            print(
                "Joint invalido."
            )

            return


        # ====================================================
        # POSICION ACTUAL
        # ====================================================

        actual = self.sim.getJointPosition(
            joint
        )


        # ====================================================
        # TARGET ACTUAL
        # ====================================================

        try:

            target_actual = (
                self.sim.getJointTargetPosition(
                    joint
                )
            )

        except Exception:

            target_actual = actual


        # Si el target esta muy lejos de la posicion real,
        # usamos la posicion real como referencia.
        if abs(
            target_actual - actual
        ) > math.radians(15):

            referencia = actual

        else:

            referencia = target_actual


        # ====================================================
        # LIMITES
        # ====================================================

        minimo, maximo = self.limites[
            numero_joint
        ]


        # ====================================================
        # NUEVO DESTINO
        # ====================================================

        destino = (
            referencia
            + direccion
            * self.incremento
        )


        destino = max(
            minimo,
            min(
                maximo,
                destino
            )
        )


        print()
        print(
            "=========================================="
        )

        print(
            f"MOVIENDO J{numero_joint}"
        )

        print(
            "=========================================="
        )

        print(
            f"Actual: "
            f"{math.degrees(actual):.1f} grados"
        )

        print(
            f"Target anterior: "
            f"{math.degrees(target_actual):.1f} grados"
        )

        print(
            f"Nuevo target: "
            f"{math.degrees(destino):.1f} grados"
        )


        # ====================================================
        # ENVIAR TARGET
        # ====================================================

        self.sim.setJointTargetPosition(
            joint,
            destino
        )


        # ====================================================
        # ESPERAR A QUE EL JOINT SE ACERQUE
        # ====================================================

        tiempo_inicio = time.time()

        timeout = 3.0

        tolerancia = math.radians(
            1.0
        )


        while True:

            posicion = self.sim.getJointPosition(
                joint
            )


            error = abs(
                destino - posicion
            )


            if error <= tolerancia:

                break


            if (
                time.time()
                - tiempo_inicio
                > timeout
            ):

                break


            time.sleep(
                0.03
            )


        # ====================================================
        # VERIFICAR
        # ====================================================

        posicion_final = (
            self.sim.getJointPosition(
                joint
            )
        )


        print(
            f"Posicion final leida: "
            f"{math.degrees(posicion_final):.1f} grados"
        )


        diferencia = abs(
            posicion_final
            - destino
        )


        if diferencia <= tolerancia:

            print(
                "Movimiento confirmado."
            )

        else:

            print(
                "ADVERTENCIA: "
                "el joint no alcanzo el target."
            )


    # ========================================================
    # GRIPPER
    # ========================================================

    def toggle_gripper(self):

        self.gripper_abierto = (
            not self.gripper_abierto
        )


        if self.gripper_abierto:

            valor = 1

            estado = "ABRIENDO"

        else:

            valor = 0

            estado = "CERRANDO"


        self.sim.setIntProperty(
            self.sim.handle_scene,
            "signal.RG2_open",
            valor
        )


        print()
        print(
            "=========================================="
        )

        print(
            f"GRIPPER: {estado}"
        )

        print(
            "=========================================="
        )


    # ========================================================
    # MOSTRAR POSICIONES
    # ========================================================

    def mostrar_posiciones(self):

        q1 = self.sim.getJointPosition(
            self.J1
        )

        q2 = self.sim.getJointPosition(
            self.J2
        )

        q3 = self.sim.getJointPosition(
            self.J3
        )


        print()
        print("POSICIONES ACTUALES:")

        print(
            f"J1 = {math.degrees(q1):.1f} grados"
        )

        print(
            f"J2 = {math.degrees(q2):.1f} grados"
        )

        print(
            f"J3 = {math.degrees(q3):.1f} grados"
        )


    # ========================================================
    # CERRAR
    # ========================================================

    def cerrar(self):

        print()

        print(
            "Deteniendo simulacion..."
        )


        try:

            self.sim.stopSimulation()

        except Exception:

            pass


        time.sleep(
            0.5
        )


        print(
            "Control del robot finalizado."
        )