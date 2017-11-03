# PyTorch Project Demo 
This is the PyTorch code for the project of the visual-auditory class:

## Requirements
* [PyTorch](http://pytorch.org/)

* torchvision

```
suggest you to install the packages following the official website
```

## Before train your own model

change the *data_dir* and *flist* in [configs/train_config.yaml](https://github.com/uzeful/VA_Project/blob/master/proj_demo/configs/train_config.yaml) to your data path and filename list

## Before evaluate your trained model

* change the *data_dir*, *video_flist* and *audio_flist* in [configs/test_config.yaml](https://github.com/uzeful/VA_Project/blob/master/proj_demo/configs/test_config.yaml) to your data path and filename list

* add the path of your trained model after the key *init_model*

## train your model
```
python train.py    
```

## evaluate your model    
```
python evaluate.py
```
