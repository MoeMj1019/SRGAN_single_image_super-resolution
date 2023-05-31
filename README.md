# SRGAN_single_image_super-resolution

## Introduction
This is a Pytorch implementation for Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.

- you can train models on your own datasets
- you can test your models on your own datasets
- you can use the pretrained model provided in the pretrained_models folder
- you can use any any images for testing, just make sure to change the path in the code accordingly 

## Environment Setup
Open anaconda prompt and cd to the folder where you have your environment.yml file
- uncomment the line of the environment_name you want to create according to your resources (GPU or CPU)

in terminal run the following command:
- conda env create -f environment.yml

- conda activate srganenv_gpu
or 
- conda activate srganenv_gpu

Now Install Pytorch as per your resources

#### GPU
- conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

##### CPU Only
- conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch


now you can either run the code in jupyter notebook or in terminal
## dataset
we are using a small dataset of 100 images, 90 training and 10 testing.
feel free to use any dataset you want, just make sure to change the path in the code accordingly

## Run the code in Jupyter Notebook
open jupyter notebook and go throught the code cell by cell, there you will find the instructions/explenations for each cell

## Run the code in Terminal

#### Train your Model:
python main.py --mode train --LR_path train_data/realistic_nature_100/train_LR/ --GT_path train_data/realistic_nature_100/train_LR/
- this will run the code with default parameters and will save the model in ./model/XXXX.pt
- take a look at the main.py file to see the rest of the parameters and their default values in case you want to change them
    ei. set smaller pre_train_epoch and fine_train_epoch if you want to train the model for shorter time

#### Test your Model:
##### test with psnr results and optional saved output images :
python main.py --mode test --LR_path test_data/realistic_nature_100/test_LR/ --GT_path train_data/realistic_nature_100/train_LR/ --generator_path ./model/XXXXX.pt
- change the XXXX to a valid model (generator) name
otherwise you can use the pretrained model provided in the pretrained_models folder:
python main.py --mode test --LR_path test_data/realistic_nature_100/test_LR/ --GT_path train_data/realistic_nature_100/train_LR/ --generator_path ./pretrained_models/SRGAN.pt

##### test with only saved output images :
python main.py --mode test --LR_path test_data/realistic_nature_100/test_LR/ --generator_path ./model/XXXXX.pt
- change the XXXX to a valid model (generator) name
otherwise you can use the pretrained model provided in the pretrained_models folder:
python main.py --mode test --LR_path test_data/realistic_nature_100/test_LR/ --generator_path ./pretrained_models/SRGAN.pt