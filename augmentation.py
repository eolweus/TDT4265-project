import albumentations as A
import cv2
import numpy as np
from PIL import Image

class Augmenter():
    def __init__(self):
        self.transform = A.Compose(
            [
                A.RGBShift(r_shift_limit=25, b_shift_limit=25, g_shift_limit=25, p=0.7),
                A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                A.OneOf([
                    A.GaussianBlur(p=0.5),
                    A.Blur(p=0.5)
                ], p=1.0)
            ] 
        )

        self.rotate_TEE = A.Compose(
            [
                A.Rotate(limit=[90,90], border_mode=cv2.BORDER_CONSTANT, p=1.0)
                # A.RandomRotate90(factor=1)
            ]
        )

    # TODO: check that augmentations work
    def augment_image(self, image, mask):
        augmentations = self.transform(image=image, mask=image)
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask

    def rotate_image90(self, image, mask):
        augmentations = self.rotate_TEE(image=image, mask=image)
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask