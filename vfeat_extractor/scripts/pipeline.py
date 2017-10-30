"""Read TFRecord and download videos and extract features."""

import argparse
import os

import pafy
import tensorflow as tf
import torch

import init_path
from misc import config as cfg
from misc.reader import Reader
from misc.utils import (concat_feat_var, get_dataloader, make_cuda,
                        make_variable)
from models import inception_v3


def is_valid(video):
    """Check if the video is available and longer than 3 minutes."""
    return video.length != -1


def download(video, save_dir, vid):
    """Download videos whose urls are stored in TFRecord from Youtube-8M."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("--> downloading {}".format(video.title))

    best = video.getbest(preftype="mp4")
    filename = best.download(
        filepath=os.path.join(save_dir,
                              "{}.{}".format(vid, best.extension)))
    print("--> saved to {}".format(filename))

    return os.path.join(save_dir, "{}.{}".format(vid, best.extension))


def extract(model, filepath, vid):
    """Extract features by inception_v3."""
    # data loader for frames in ingle video
    data_loader = get_dataloader(dataset="VideoFrame",
                                 path=filepath,
                                 num_frames=cfg.num_frames,
                                 batch_size=cfg.batch_size)
    # extract features by inception_v3
    feats = None
    for step, frames in enumerate(data_loader):
        print("--> extract features [{}/{}]".format(step + 1,
                                                    len(data_loader)))
        feat = model(make_variable(frames))
        feats = concat_feat_var(feats, feat.data.cpu())

    print("--> save feats to {}"
          .format(cfg.inception_v3_feats_path.format(vid)))
    torch.save(feats, cfg.inception_v3_feats_path.format(vid))
    print("--> delete original video file: {}".format(filepath))
    os.remove(filepath)


if __name__ == '__main__':
    model = make_cuda(inception_v3(pretrained=True,
                                   transform_input=True,
                                   extract_feat=True))
    model.eval()

    # record_root = cfg.record_root
    record_root = "/media/m/E/yt8m_video_level/train/"
    for record_file in os.listdir(record_root):
        if os.path.splitext(record_file)[1] != ".tfrecord":
            continue
        iterator = tf.python_io.tf_record_iterator(
            os.path.join(record_root, record_file))
        for record in iterator:
            result = Reader(record)
            print("Processing {}".format(result.vid))
            # skip existed files
            v_path = os.path.join(cfg.video_root,
                                  "{}.mp4".format(result.vid))
            f_path = cfg.inception_v3_feats_path.format(result.vid)
            if os.path.exists(v_path) or os.path.exists(f_path):
                print("--> skipping {}".format(result.vid))
                continue
            try:
                # get video info
                url = "https://www.youtube.com/watch?v={}".format(result.vid)
                video = pafy.new(url)
                if is_valid(video):
                    # download video
                    video_file = download(video, cfg.video_root,
                                          result.vid)
                    # extract features
                    extract(model, video_file, result.vid)
            except:
                print("--> error occurs! skipping {}".format(result.vid))
                continue
