import torch
import torch.nn as nn


class FingerCNN(nn.Module):

    def __init__(self, num_classes=5):
        super(FingerCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (4, 4)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            64 * 4 * 4,
            128
        )

        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(
            p=0.30
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":

    modelo = FingerCNN(
        num_classes=5
    )

    entrada = torch.randn(
        1, 3, 128, 128
    )

    salida = modelo(
        entrada
    )

    total_parametros = sum(
        p.numel()
        for p in modelo.parameters()
    )

    parametros_entrenables = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print("==========================================")
    print("ARQUITECTURA CNN")
    print("==========================================")

    print(modelo)

    print("\nEntrada:")
    print(entrada.shape)

    print("\nSalida:")
    print(salida.shape)

    print("\nNumero de clases:")
    print(5)

    print("\nParametros totales:")
    print(total_parametros)

    print("\nParametros entrenables:")
    print(parametros_entrenables)