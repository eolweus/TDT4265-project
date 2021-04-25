# TDT4265-project
The repository for the project in TDT4265 the spring of 2021 by group 3

# Datasyn Final Project

This Project has been developed by:
```
Gulleik L Olsen         gulleik@hotmail.com           Software Engineer
Erling Olweus           erling.olweus@gmail.com       Software Engineer
Fredrik Gran-Jansen     fredrik.granjansen@gmail.com  Software Engineer
```

## Project description
In this project we will go through a typical medical imaging workflow often seen in real world
scenarios. The main task in the project will be segmentation of structures of interest in ultrasound
images of the heart (see figure). Two modalities will be considered, transthoracic ultrasound
(with ground truth segmentations available) and trans esophageal ultrasound (expert validated
test set is available). 

We were free to implement our solution however we wanted, but there were some system requirements which had to be fulfilled:

- **DICE score higher than baseline code for TTE.**

    Implement a dice score (do it yourself or use one from a python package) for the sample code (that will be considered the baseline of 90%). Estimate a dice score for the      left ventricle (LV), compare it to your work to it, describe what you did to improve the dice.
- **Model tested and working on TEE images.**

     Test your trained TTE model on the 19 TEE images with ground truth (you can discard the h5 files they were given as sample until the other ones were ready).
- **Modify the network to be able to segment all 3 classes**

  The classes are the left ventricle, myocardium and left atrium on the original CAMUS data. Recommended test set is patients 401-450.
- **Rotate the TEE data by 90 degrees in order to make it more similar to TTE data**

  Generate TTE and TEE images that are most similar content wise
- **Convert the data to iso-tropic pixel size**

  The element spacing should be equal in both directions.
- **Try and report Dice score on various input resolutions of your data**

  Subsample or oversample the images and the ground truth and then see what the network achieves in terms of dice score. 
- **Use both the ED and ES timepoints from the TTE data for training purposes**

  This is regarding the CAMUS dataset where you need to use all 4 images per patient (4CH and 2CH both at ED and ES).
- **Report the effect of image smoothing on the Dice score (Gaussian or bilateral smoothing are 2 possible options)**

  Use some sort of smoothing filter on the image.
- **Surprise us with an idea you have**

  This is an idea that you have/want to try out that is not part of the requirements.

### Download the starter code

Clone this repostiory:

```bash
git clone https://github.com/gg1977/Unet2D
```

You can also download this repository as a zip file and unzip it on your computer.

## Setup

To further develop this project make sure to install python-decouple and add a .env file containing the path to your training images.
Your training images should be seperated into the following structure

| CAMUS_resized
|---train_gray
|---train_gt

.env file should be in the root folder of the project and should look like this:

IMAGE_BASE_PATH=your_image_base_path

for cybelle this is:
../../../../work/datasets/medical_project/CAMUS_resized

for local Erling uses:
./home/gkiss/Data/CAMUS_resized

## Prerequisites

### Packages

To install the python-decouple package, SimpleITK, albumentations for augmentation, PIL, torch, please use the following respective commands

```
conda install -c conda-forge python-decouple
pip install SimpleITK
pip install -U albumentations
python3 -m pip install --upgrade Pillow
conda install PyTorch -c PyTorch
```
 Then, you can train the model on cityscapes by running the file “train.py”.
 
### Setup
Input in TTE loader is a directory where all the patient sub directories are. Take the standard setup from the challenge and make a seperate test folder that contains the 50 medical images from patient 401-450. 

We are using a .env file to keep track of file paths. See at the top of train.py the respectful paths and please implement these.


## Our system

### DatasetLoader
Loads data from TTE, TEE and CAMUS. Creates a dictionary with the paths to TTE and TEE images with corresponding ground truth image from the CAMUS dataset. Combines the ground truth and gray files. Reads the images with SimpleITK, creates an array containing the images and opens mask file.

### Unet2d
Architecture with 4 down and 4 up convolutional layers. The contraction blocks and extraction blocks each have 2 convolutional layers, each followed by a BatchNorm2d and a ReLU function. The contraction blocks ends with a MaxPool2d function, while the extraction blocks ends with a ConvTranspose2d function.

### augmentation
We use albumentation.ai for augmentation, such as Rotate(), GaussianBlur() and Blur().

### train
A collection of everything corresponding to training such as logging, outputs and checkpoints.

### trainer
The logic behind training the model such as running through epochs. This module does all the traning.

### logs

 
 
