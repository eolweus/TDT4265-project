import albumentations as A
import cv2
import numpy as np
from PIL import Image
from configs import cfg

class Augmenter():
    def __init__(self):
        self.transform = A.Compose(
            [   
                A.RandomResizedCrop(*cfg.INPUT.IMAGE_SIZE, scale=(0.3, 1.0), p=8.0),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.8),
                A.GaussianBlur(p=8.0)
            ] 
        )
        
        self.transformations = {
            "Transform": self.transform,
            "Gaussian_blur": A.GaussianBlur(p=1.0),
            "Blur": A.Blur(p=1.0),
        }

    def rotate_image(self, image, mask, degrees=180):
        transform = A.Rotate(limit=(degrees, degrees), border_mode=cv2.BORDER_CONSTANT, p=1.0)
        augmentations = transform(image=image, mask=mask)
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]
        return aug_image, aug_mask

    def transform_image(self, image, mask, aug_choice="Transform"):
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
        
