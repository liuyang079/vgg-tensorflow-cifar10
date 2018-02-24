# vgg-tensorflow-cifar10
cifar10 dataset in vgg architecture by tensorflow
## Hardware config info.
tensorflow-gpu: 1.4  
python: 3.5  
gpu: tesla P4  
system: windows7(main) or ubuntu16.04  
IDE: pycharm  
## Introduction
This project includes four sessions:  
data processing: cifar10_vgg_processing.py  
network architecture: vgg.py  
model train, validation and test: cifar10_vgg_train  
hyper parameters: hyper_parameters.py  
For the data processing section, cifar10 dataset, which is converted to filepath form, it looks like TF-record form. you need to convert initial binary cifar10 data into .np file, and main purpose is to make our own datsets. In addition, read batch-file-path can reduce the memory space for running loads, for example ImageNet datasets.  
For the vgg_train file, which includes trian model, validation model and test model.  
## Validation errors
The lowest valdiation errors of vgg19 is 8.7% . You can change the number of the total layers by changing  vgg.py. vgg19 model 
use conv-bn-relu layer and dropout layer.

Network | Lowest Validation Error | Max_step | batch_size
------- | ----------------------- | -------- | -----------
vgg-19 |        8.7%              |   40000  |  128






