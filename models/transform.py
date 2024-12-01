import torch
from torchvision.transforms import v2 as T


class ImageAugmentation:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.RandomPerspective(0.2),
                T.RandomChoice([T.GaussianBlur(k) for k in [3, 5, 7]]),
                T.ColorJitter(brightness=(0.7, 1.3), contrast=(0.9, 1.1)),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class PILToNormalizedTensor:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
