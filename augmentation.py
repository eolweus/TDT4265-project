import albumentations as A
import cv2
import numpy as np
from PIL import Image
from configs import cfg

class Augmenter():
    def __init__(self):
        self.transform = A.Compose(
            [   
                # A.CenterCrop(100, 100),
                A.RandomResizedCrop(*cfg.INPUT.IMAGE_SIZE, scale=(0.2, 1.0), p=1.0),
                # A.RandomResizedCrop(*cfg.INPUT.IMAGE_SIZE, scale=(0.2, 1.0), p=0.5),
                # A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                A.OneOf([
                    A.GaussianBlur(p=0.5),
                    A.Blur(p=0.5)
                ], p=0.5)
            ] 
        )

        # TODO: check if this is the right rotation
        self.rotate_TEE = A.Compose(
            [
                A.Rotate(limit=[90,90], border_mode=cv2.BORDER_CONSTANT, p=1.0)
        )
        
        self.transformations = {
            "Transform": self.transform,
            "Rotate90": self.rotate_TEE,
            "Gaussian_blur": A.GaussianBlur(p=1.0),
            "Blur": A.Blur(p=1.0),
            "RGB_shift": A.RGBShift(r_shift_limit=25, b_shift_limit=25, g_shift_limit=25, p=0.7)
        }

    def augment_image(self, image, mask):
        augmentations = self.transform(image=image, mask=mask)
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask

    def rotate_image90(self, image, mask):
        augmentations = self.rotate_TEE(image=image, mask=image)
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask

    def blur(self, image, mask, gaussian_blur=True):
        if gaussian_blur:
            augmentations = A.GaussianBlur(p=1.0)
        else:
            augmentations = A.Blur(p=1.0)
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask

    def transform_image(self, image, mask, aug_choice):
        assert aug_choice in self.transformations.keys()\
        , print("ASSERTION ERROR: Input aug_choice is not an option for transform image.\n",
                "Valid options are: ", self.transformations.keys())
        augmentations = self.transformations[aug_choice]
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask

    def crop_and_resize(self, image, mask, min_maxes):
        transform = A.Compose(
            [
                A.Crop(*min_maxes),
                A.RandomResizedCrop(*cfg.INPUT.IMAGE_SIZE, scale=(1.0, 1.0), p=1.0),
            ]
        )
        augmentations = transform(image=image, mask=mask)
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask