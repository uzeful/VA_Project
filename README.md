## Aim of This Project

This project is set up to assist you to finish your final project of the class "Introduction to Visual-Auditory Information System".

## Introduction

[afeat_extractor](https://github.com/uzeful/VA_Project/tree/master/afeat_extractor) and [vfeat_extractor](https://github.com/uzeful/VA_Project/tree/master/vfeat_extractor) are respectively used to extract the visual and auditory features of the video. Specifically, in our project, we extract the 128d auditory feature and 1024d visual feature every second, and we totally extract 120 seconds of features. Therefore, every video corresponds to 120×128 auditory feature and 120×1024 visual feature, which are respectively saved as the numpy compressed format (\*.npy).

* The audio feature is extracted by a Vgg-like CNN model (implemented in tensorflow).

* The visual feature is extracted by the inception v3 model (implemented in pytorch).

## Usage

* [afeat_extractor/infer_folder_afeat.py](https://github.com/uzeful/VA_Project/blob/master/afeat_extractor/infer_folder_afeat.py) is used to extract the auditory features of the videos *in your defined folder*.

* [vfeat_extractor/infer_folder_vfeat.py](https://github.com/uzeful/VA_Project/blob/master/vfeat_extractor/infer_folder_vfeat.py) is used to extract the visual features of the videos *in your defined folder*.

Before using them to extract features, you should firstly download the [pretrained vggish models](http://pan.baidu.com/s/1nuVq3PZ) and [pretrained inception models](http://pan.baidu.com/s/1dEV6J41), and then respectively put them under the folder "afeat_extractor/" and folder "vfeat_extractor/pretrained/". 

Moreover, you should also install the required dependencies, such as pytorch and tensorflow. The detailed requirements can be found in the subfolders "afeat_extractor" and "vfeat_extractor".

## Demo

[proj_demo](https://github.com/uzeful/VA_Project/tree/master/proj_demo) provides one simple example to learn the similarity metric between the 120×1024 visual feature and 120×128 auditory feature. *Note: the provided demo was implemented in pytorch.*

[proj_demo/model2.py](https://github.com/uzeful/VA_Project/blob/master/proj_demo/model2.py) contains our simply designed model.

[proj_demo/train.py](https://github.com/uzeful/VA_Project/blob/master/proj_demo/train.py) contains training code.

[proj_demo/evaluate.py](https://github.com/uzeful/VA_Project/blob/master/proj_demo/evaluate.py) contains testing code.

## Acknowlegdements

* The original implementation of the visual feature extractor could be found from [this link](https://github.com/corenel/yt8m-feature-extractor).

* The original implementation of the auditory feature extractor could be found from [this link](https://github.com/tensorflow/models/tree/master/research/audioset).

## Q&A

If you have any question, just add a new issue!
