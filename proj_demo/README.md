# PyTorch Project Demo 
This is the PyTorch code for the project of the visual-auditory class:

## Requirements
* [PyTorch](http://pytorch.org/)

* [torchvision](https://github.com/pytorch/vision)

```
suggest you to install the packages following instructions from the official website
```

## Before training your own model

change the *data_dir* and *flist* in [configs/train_config.yaml](https://github.com/uzeful/VA_Project/blob/master/proj_demo/configs/train_config.yaml) to your data path and filename list

## Before evaluating your trained model

* change the *data_dir*, *video_flist* and *audio_flist* in [configs/test_config.yaml](https://github.com/uzeful/VA_Project/blob/master/proj_demo/configs/test_config.yaml) to your data path and filename list

* add the path of your trained model after the key *init_model*

## Train your model
```
python train.py    
```

## Evaluate your model
```
python evaluate.py
```

## Self-defined model
```
modify model.py to configure your own model, and change the hyper-parameters in the config files (configs/train_config.yaml)
```

## Tested Environments
Ubuntu16.04+python2.7+torch0.2.0/0.3.0+GPU/CPU    
Ubuntu16.04+python3.6+torch0.2.0/0.3.0+GPU/CPU    
