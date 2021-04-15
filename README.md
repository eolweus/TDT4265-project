# TDT4265-project
The repository for the project in TDT4265 the spring of 2021 by group 3


### Setup

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


To install the python-decouple package please use the following command

conda install -c conda-forge python-decouple
