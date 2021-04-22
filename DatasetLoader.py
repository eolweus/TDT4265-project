import numpy as np
import torch

#os: provides functions for interacting with the operating system. 
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image, ImageOps #bruker vi fortsatt ImageOps?
#SimpleITK: Open-source multi-dimensional image analysis in Python
import SimpleITK as sitk

#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, data_dir, pytorch=True):
    """
    Args:
        data_dir: Directory including both gray image directory and ground truth directory.
    """
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = self.create_dict(data_dir)
        self.pytorch = pytorch

    def create_dict(self, data_dir):
    """
    Args:
        data_dir: Directory including both gray image directory and ground truth directory.
    """
        return
        
    def combine_files(self, gray_file: Path, gt_dir):
        #combines gray and ground truth files
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
       
                                       
    def __len__(self):
        #length of all files to be loaded
        return len(self.files)
     
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
        raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        return x, y
    
    def get_as_pil(self, idx): #fjernes?
        #get an image for visualization
        # arr = 256*self.open_as_array(idx)
        arr = 256*self.open_mask(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')


class ResizedLoader(DatasetLoader):
    def __init__(self, data_dir, pytorch=True):
        """
            Is called when model is initialized.
            Args:
                data_dir: Directory including both gray image directory and ground truth directory.
        """
        super().__init__(data_dir)

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
        raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        # arr = 256*self.open_as_array(idx)
        arr = 256*self.open_mask(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')



class TTELoader(DatasetLoader):
    def __init__(self, data_dir, pytorch=True):
        super().__init__(data_dir)
        
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
        raw_us = np.stack([self.load_itk(self.files[idx]['gray'])], axis=2)
        # raw_us = np.array([self.load_itk(self.files[idx]['gray'])])

        
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)

    # TODO: edit this to fit tte
    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = (self.load_itk(self.files[idx]['gt']))
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

