import numpy as np
import torch

#os: provides functions for interacting with the operating system. 
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image, ImageOps #bruker vi fortsatt ImageOps?
#SimpleITK: Open-source multi-dimensional image analysis in Python
import SimpleITK as sitk
import albumentations as A
import cv2
from configs import cfg

from augmentation import Augmenter

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, data_dir, use_transforms=False, pytorch=True):
        """
        Args:
            data_dir: Directory including both gray image directory and ground truth directory.
        """
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = self.create_dict(data_dir)
        self.pytorch = pytorch
        self.use_transforms = use_transforms
        self.augmenter = Augmenter()

    def create_dict(self, data_dir):
        """
        Args:
            data_dir: Directory including both gray image directory and ground truth directory.
        """
        return
        
    def combine_files(self, gray_file: Path, gt_dir):
        return
                                       
    def __len__(self):
        #length of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        return
    
    def open_mask(self, idx, add_dims=False):
        return
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        img_as_array = self.open_as_array(idx, invert=self.pytorch)
        mask_as_array = self.open_mask(idx, add_dims=False)

        if self.use_transforms:
            img_as_array, mask_as_array = self.augmenter.augment_image(image=img_as_array, mask=mask_as_array)

        # squeeze makes sure we get the right shape for the mask
        x = torch.tensor(img_as_array, dtype=torch.float32)
        y = torch.tensor(np.squeeze(mask_as_array), dtype=torch.torch.int64)

        return x, y

    
    def get_as_pil(self, idx): #fjernes?
        #get an image for visualization
        arr = 256*self.open_mask(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')


class ResizedLoader(DatasetLoader):
    def __init__(self, data_dir, use_transforms=False, pytorch=True):
        """
            Args:
                data_dir: Directory including both gray image directory and ground truth directory.
        """
        super().__init__(data_dir, use_transforms=use_transforms)

    def create_dict(self, data_dir):
        """
            Loops through the files and combines both gray images and ground truth to a directory.
            Args:
                data_dir: Directory including both gray image directory and ground truth directory.
            Returns:
                Directory containing both gray images and ground truth.
        """
        gray_dir, gt_dir = [Path(os.path.join(data_dir, name)) for name in next(os.walk(data_dir))[1]] 
        return [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        
    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                           ], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)

    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        print(raw_mask.shape)
        raw_mask = np.where(raw_mask>100, 1, 0)
        print(raw_mask.shape)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask


class TTELoader(DatasetLoader):
    def __init__(self, data_dir, use_transforms=False, pytorch=True):
        # self.img_size = 384 # må gjøre dette dynamisk
        self.img_size = cfg.INPUT.IMAGE_SIZE
        super().__init__(data_dir, use_transforms=use_transforms)    
 
    def create_dict(self, data_dir):
        """
        Generates dictionary with paths to TTE images with corresponding ground truth image from the CAMUS dataset.
        Paths into patient folder and walks through the patients and finds images on type: (['2CH_ED', '2CH_ES','4CH_ED','4CH_ES'])
        Args:
            data_dir: Directory including both gray image directory and ground truth directory.
        Return:
            dict.list: List containing path to the medical image data and the ground truth data
        """
        dict_list=[]
        for root, dirs, files in os.walk(data_dir, topdown=True):
            for name in files:
                if name[-7:] == "_gt.mhd":
                    dict_list.append(self.combine_files(root, name))
        return dict_list  


    def combine_files(self, root, gt_file_name):
        """
        Match path to ground truth data and remove _gt from gray image path.
        Args:
            root: beginning of the folder structure
            gt_file_name: name of the ground truth data
        Return:
            files: dictionary with values for the path to ground truth and gray image data
        """
        gt_path = os.path.join(root, gt_file_name)
        files = {'gt': gt_path, 
                'gray': gt_path.replace("_gt", "")}
        return files

    # TODO: make sure all the images are of equal size
    def open_as_array(self, idx, invert=False):
        raw_us = np.array(self.load_itk(self.files[idx]['gray']))
        raw_us = cv2.resize(raw_us, dsize=self.img_size)
        raw_us = np.stack([raw_us], axis=2)
        
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)

    # TODO: edit this to fit tte
    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array((self.load_itk(self.files[idx]['gt'])))
        raw_mask = cv2.resize(raw_mask, dsize=self.img_size)
        # raw_mask = np.where(raw_mask>100, 1, 0)
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask


    def load_itk(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        img_as_array = sitk.GetArrayFromImage(itkimage)
        # Removes the z-axis and transposes the image
        processed_img_as_array = np.transpose(np.squeeze(img_as_array))
        return processed_img_as_array
    

class TEELoader(DatasetLoader):
    def __init__(self, data_dir, use_transforms=False, pytorch=True):
        self.img_size = 384 # må gjøre dette dynamisk
        super().__init__(data_dir, use_transforms=use_transforms)        
    
    def create_dict(self, data_dir, Tif_dir=False):
        dict_list=[]
        for root, dirs, files in os.walk(data_dir, topdown=True):
            for name in files:
                 if name[:5] == "gray_":
                    dict_list.append(self.combine_files(root, name))
        return dict_list  

    def combine_files(self, root, gt_file_name):
        gray_path = os.path.join(root, gt_file_name)
        gt_path = gray_path.replace("gray_", "gt_gt_")
        gt_path = gt_path.replace(".jpg", ".tif")
        gt_path = gt_path.replace("train_gray", "train_gt")
        files = {'gt': gt_path, 
                'gray': gray_path}
        return files

    def open_as_array(self, idx, invert=False):
        raw_us = np.array(Image.open(self.files[idx]['gray']))
        print("image:",raw_us.shape)
        raw_us = cv2.resize(raw_us, dsize=(self.img_size, self.img_size))
        raw_us = np.stack([raw_us], axis=2)
        print("image:",raw_us.shape)
        
        
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)

    def open_mask(self, idx, add_dims=False):
        #open mask file
        # raw_mask = np.array((cv2.imread(self.files[idx]['gt'], cv2.IMREAD_GRAYSCALE)))
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = cv2.resize(raw_mask, dsize=(self.img_size, self.img_size))

        # print(np.unique(raw_mask, return_counts = True))
        # TODO: check if this works, im not sure if it does
        raw_mask = np.where(raw_mask>100, raw_mask, 0)
        raw_mask = np.where(raw_mask>200, 2, raw_mask)
        raw_mask = np.where(raw_mask>100, 1, raw_mask)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    # Test function
    def open_mask_2(self, idx, add_dims=False):
        #open mask file
        # raw_mask = np.array((cv2.imread(self.files[idx]['gt'], cv2.IMREAD_GRAYSCALE)))
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        # print(raw_mask.shape)
        # raw_mask = np.delete(raw_mask, 1)
        raw_mask = cv2.resize(raw_mask, dsize=(self.img_size, self.img_size))
        # print(raw_mask.shape)
        # raw_mask = np.stack([raw_mask], axis=2)
        # print(raw_mask.shape)

        # print(np.unique(raw_mask, return_counts = True))
        # TODO: check if this works, im not sure if it does
        raw_mask = np.where(raw_mask>100, raw_mask, 0)
        raw_mask = np.where(raw_mask>200, 2, raw_mask)
        raw_mask = np.where(raw_mask>100, 1, raw_mask)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask


    def __getitem__(self, idx):
        #get the image and mask as arrays
        img_as_array = self.open_as_array(idx, invert=self.pytorch)
        mask_as_array = self.open_mask(idx, add_dims=False)
        img_as_array, mask_as_array = self.augmenter.rotate_image90(image=img_as_array, mask=mask_as_array)

        if self.use_transforms:
            img_as_array, mask_as_array = self.augmenter.augment_image(image=img_as_array, mask=mask_as_array)

        x = torch.tensor(img_as_array, dtype=torch.float32)
        # squeeze makes sure we get the right shape for the mask
        y = torch.tensor(np.squeeze(mask_as_array), dtype=torch.torch.int64)

        return x, y
    
    def remove_image_borders(self, idx):
        img_as_array = self.open_as_array(idx)
        # print(img_as_array.shape)
        mask_as_array = self.open_mask(idx, add_dims=False)

        x_max, y_max, y_min = self.find_max_min_values(img_as_array)
        min_maxes = (0, y_min, x_max, y_max)
        print(min_maxes)
        img, mask = self.augmenter.crop_and_resize(img_as_array, mask_as_array, min_maxes)
        return img, mask

    # TODO: fix this so that y_max and y_min are correct
    def find_max_min_values(self, img_as_array):
        img_as_array= np.squeeze(img_as_array)
        x_max = y_max = 0
        y_min = 1000
        y_max_set = False
        for y in range(cfg.INPUT.IMAGE_SIZE[1]):
            for x in range(cfg.INPUT.IMAGE_SIZE[0]):
                if img_as_array[y][x] != 0:
                    if x_max < x:
                        x_max = x
                    if not y_max_set:
                        y_max = cfg.INPUT.IMAGE_SIZE[1] - y
                    if y_max < y:
                        y_max = y
        return x_max, y_max, y_min
        
                    

