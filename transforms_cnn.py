from torchvision import transforms

IMG_SIZE = 128


def quitar_borde(imagen):
    borde = 4

    return imagen.crop(
        (
            borde,
            borde,
            imagen.width - borde,
            imagen.height - borde
        )
    )


train_transform = transforms.Compose([
    transforms.Lambda(quitar_borde),

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.RandomRotation(
        degrees=12,
        fill=255
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10),
        fill=255
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


eval_transform = transforms.Compose([
    transforms.Lambda(quitar_borde),

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])