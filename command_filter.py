import time
from collections import deque, Counter


class CommandFilter:

    def __init__(
        self,
        window_size=7,
        min_votes=5,
        confidence_threshold=0.80,
        cooldown=0.8
    ):

        self.window_size = window_size
        self.min_votes = min_votes
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown

        self.history = deque(
            maxlen=window_size
        )

        self.last_command_time = 0.0

        # Clase estable detectada actualmente
        self.stable_class = None

        # Ultimo comando realmente enviado
        self.last_sent_command = None


    def reset(self):

        self.history.clear()

        self.stable_class = None

        self.last_sent_command = None


    def update(
        self,
        predicted_class,
        confidence
    ):

        # ====================================================
        # 1. RECHAZAR PREDICCIONES DE BAJA CONFIANZA
        # ====================================================

        if confidence < self.confidence_threshold:

            self.history.append(None)

            return None


        # ====================================================
        # 2. GUARDAR PREDICCION
        # ====================================================

        self.history.append(
            predicted_class
        )


        # ====================================================
        # 3. ESPERAR A LLENAR LA VENTANA
        # ====================================================

        if len(self.history) < self.window_size:

            return None


        # ====================================================
        # 4. QUITAR PREDICCIONES INVALIDAS
        # ====================================================

        votos_validos = [
            clase
            for clase in self.history
            if clase is not None
        ]


        if len(votos_validos) == 0:

            self.stable_class = None

            return None


        # ====================================================
        # 5. VOTACION MAYORITARIA
        # ====================================================

        contador = Counter(
            votos_validos
        )

        clase_mayoritaria, votos = (
            contador.most_common(1)[0]
        )


        # ====================================================
        # 6. EXIGIR MINIMO DE VOTOS
        # ====================================================

        if votos < self.min_votes:

            self.stable_class = None

            return None


        # ====================================================
        # 7. CLASE ESTABLE
        # ====================================================

        self.stable_class = (
            clase_mayoritaria
        )


        # ====================================================
        # 8. CLASE 0 = REPOSO / SIN COMANDO
        # ====================================================

        if clase_mayoritaria == 0:

            # Permite volver a enviar el mismo comando
            # después de regresar a 0.
            self.last_sent_command = None

            return None


        # ====================================================
        # 9. EVITAR REPETIR EL MISMO COMANDO
        #    MIENTRAS SE MANTIENE EL MISMO GESTO
        # ====================================================

        if (
            self.last_sent_command
            == clase_mayoritaria
        ):

            return None


        # ====================================================
        # 10. COOLDOWN
        # ====================================================

        tiempo_actual = time.time()

        if (
            tiempo_actual
            - self.last_command_time
            < self.cooldown
        ):

            return None


        # ====================================================
        # 11. GENERAR COMANDO
        # ====================================================

        self.last_command_time = (
            tiempo_actual
        )

        self.last_sent_command = (
            clase_mayoritaria
        )


        return clase_mayoritaria


    def get_stable_class(self):

        return self.stable_class