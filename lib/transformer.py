import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms


class Transformer:

    def __init__(self, train, augmentation_types=[]):
        width = 128
        height = 256

        transformers = []
        if train:
            if 'rotate' in augmentation_types:
                transformers.append(transforms.RandomRotation(5))
            transformers.append(transforms.Resize((height, width), interpolation=3))
            if 'crop' in augmentation_types:
                transformers.append(transforms.Pad(10))
                transformers.append(transforms.RandomCrop((height, width)))
            if 'color' in augmentation_types:
                transformers.append(transforms.ColorJitter(.4, .4, .4),)
            if 'flip' in augmentation_types:
                transformers.append(transforms.RandomHorizontalFlip())

            transformers.append(transforms.ToTensor())
            transformers.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]))
            self.transformer = transforms.Compose(transformers)
        else:
            transformers += [
                transforms.Resize((height, width), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
            self.transformer = transforms.Compose(transformers)

    def transform(self, img):
        img = self.transformer(img)

        return img
