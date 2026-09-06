import torch
import torch.nn as nn


class FingerCNNV3(nn.Module):

    def __init__(self, num_classes=5):

        super(FingerCNNV3, self).__init__()

        # ====================================================
        # BLOQUE 1
        # 3 x 128 x 128
        # ->
        # 16 x 64 x 64
        # ====================================================

        self.block1 = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            )
        )


        # ====================================================
        # BLOQUE 2
        # 16 x 64 x 64
        # ->
        # 32 x 32 x 32
        # ====================================================

        self.block2 = nn.Sequential(

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            )
        )


        # ====================================================
        # BLOQUE 3
        # 32 x 32 x 32
        # ->
        # 64 x 16 x 16
        # ====================================================

        self.block3 = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            )
        )


        # ====================================================
        # BLOQUE 4
        # 64 x 16 x 16
        # ->
        # 128 x 8 x 8
        # ====================================================

        self.block4 = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2
            )
        )


        # ====================================================
        # REDUCCION ESPACIAL
        # 128 x 8 x 8
        # ->
        # 128 x 2 x 2
        # ====================================================

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (2, 2)
        )


        # ====================================================
        # CLASIFICADOR
        # ====================================================

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            128 * 2 * 2,
            128
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            p=0.50
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )


    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":

    modelo = FingerCNNV3(
        num_classes=5
    )

    entrada = torch.randn(
        1,
        3,
        128,
        128
    )

    salida = modelo(
        entrada
    )

    total_parametros = sum(
        p.numel()
        for p in modelo.parameters()
    )

    entrenables = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print("==========================================")
    print("CNN V3 CORREGIDA")
    print("==========================================")

    print(modelo)

    print("\nEntrada:")
    print(entrada.shape)

    print("\nSalida:")
    print(salida.shape)

    print("\nParametros totales:")
    print(total_parametros)

    print("\nParametros entrenables:")
    print(entrenables)