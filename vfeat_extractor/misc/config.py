"""Config for this project."""
import os

# general
root = "/media/m/E/download/"
dataset_path = "/data/datasets/yt8m"

# config for dataset
frame_path = '/data/datasets/yt8m/movie'
frame_subdir = "frames"
video_ext = [".mp4"]
frame_num = 120
batch_size = 32

# config for extracting features from Inception v3
save_step = 10000
inception_v3_model = "pretrained/inception_v3_google-1a9a5a14.pth"

# config for PCA
n_components = 1024
pca_batch_size = 4096
pca_model = "pretrained/pca_params.pkl"
