## Target of This Repo

&#8194 This repo is set up to assist you to finish your final project of the class "Introduction to Visual-Auditory Information System". This repo mainly consists of three parts: the auditory feature extractor ( [afeat_extractor](https://github.com/uzeful/VA_Project/blob/master/afeat_extractor)), visual feature extractor ([vfeat_extractor](https://github.com/uzeful/VA_Project/blob/master/vfeat_extractor)) and also a simple project demo ([proj_demo](https://github.com/uzeful/VA_Project/tree/master/proj_demo)) which is used to predict the similarity of the audio and silent video. 

****

## Code Description

### Feature Extractors

&#8194 [afeat_extractor](https://github.com/uzeful/VA_Project/tree/master/afeat_extractor) and [vfeat_extractor](https://github.com/uzeful/VA_Project/tree/master/vfeat_extractor) are respectively used to extract the visual and auditory features of the video. Specifically, in our project, we extract the 128d auditory feature and 1024d visual feature every second, and we totally extract 120 seconds of features. Therefore, every video corresponds to 120×128 auditory feature and 120×1024 visual feature, which are respectively saved as the numpy compressed file (\*.npy).

&#8194 * The audio feature is extracted by a Vgg-like CNN model (implemented in tensorflow).

&#8194 * The visual feature is extracted by the inception v3 model (implemented in pytorch).

### How to use the feature extractors

&#8194 * [afeat_extractor/infer_folder_afeat.py](https://github.com/uzeful/VA_Project/blob/master/afeat_extractor/infer_folder_afeat.py) is used to extract the auditory features of the videos *in your defined folder*.

&#8194 * [vfeat_extractor/infer_folder_vfeat.py](https://github.com/uzeful/VA_Project/blob/master/vfeat_extractor/infer_folder_vfeat.py) is used to extract the visual features of the videos *in your defined folder*.

&#8194 Before using them to extract features, you should firstly download the [pretrained vggish models](http://pan.baidu.com/s/1nuVq3PZ) and [pretrained inception models](http://pan.baidu.com/s/1dEV6J41), and then respectively put them under the folder "afeat_extractor/" and folder "vfeat_extractor/pretrained/". 

&#8194 Moreover, you should also install the required dependencies, such as pytorch and tensorflow. The detailed requirements can be found in the subfolders "afeat_extractor" and "vfeat_extractor".


### Project Demo

&#8194 [proj_demo](https://github.com/uzeful/VA_Project/tree/master/proj_demo) provides one simple example to learn the similarity metric between the 120×1024 visual feature and 120×128 auditory feature. *Note: the provided demo was implemented in pytorch.*

&#8194 * [proj_demo/model2.py](https://github.com/uzeful/VA_Project/blob/master/proj_demo/model2.py) contains our simply designed model.

&#8194 * [proj_demo/train.py](https://github.com/uzeful/VA_Project/blob/master/proj_demo/train.py) contains training code.

&#8194 * [proj_demo/evaluate.py](https://github.com/uzeful/VA_Project/blob/master/proj_demo/evaluate.py) contains testing code.

****

## Dataset

&#8194 The provided training dataset includes 1300 video folders, each of which contains five parts:

&#8194 * frames: containing 125 video frames, where are sampled at the rate 1 frame per second    
&#8194 * \*.mp4: the video file without audio    
&#8194 * \*.wav: the audio file    
&#8194 * afeat.npy: the numpy compressed auditory feature (120\*128)    
&#8194 * vfeat.npy: the numpy compressed visual feature (120\*1024)    

&#8194 The following pic shows one example
<p align="center">
<img src="https://github.com/uzeful/VA_Project/blob/master/dataset_sample.png" width="750">
</p>

&#8194 The total dataset containing all the five parts takes about 60GB memory, and can be downloaded through the campus network. If you only use the extracted auditory feature and visual feature, then you can download the feature-only-dataset (about 150MB) from [Baidu Yun](http://pan.baidu.com/s/1qY2uyhI).

****

## Acknowlegdements

&#8194 * The original implementation of the visual feature extractor could be found from [this link](https://github.com/corenel/yt8m-feature-extractor).

&#8194 * The original implementation of the auditory feature extractor could be found from [this link](https://github.com/tensorflow/models/tree/master/research/audioset).

****

## Q&A

&#8194 If you have any question, just contact us through e-mails or add a new issue under this repo!
