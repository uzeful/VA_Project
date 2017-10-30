"""Read TFRecord and download corresponding youtube video."""

import argparse
import os

import pafy
import tensorflow as tf

import init_path
from misc import config as cfg
from misc.reader import Reader

# from youtube_dl.utils import DownloadError, ExtractorError


def is_valid(video):
    """Check if the video is available and longer than 3 minutes."""
    return video.length != -1


def download(video, save_dir, vid):
    """Download videos whose urls are stored in TFRecord from Youtube-8M."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("downloading {}".format(video.title))

    best = video.getbest(preftype="mp4")
    filename = best.download(
        filepath=os.path.join(save_dir,
                              "{}.{}".format(vid, best.extension)))
    print("saved to {}".format(filename))

    return os.path.join(save_dir, "{}.{}".format(vid, best.extension))


if __name__ == '__main__':
    record_root = cfg.record_root
    # record_root = "/media/m/E/yt8m_video_level/train/"
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
                url = "https://www.youtube.com/watch?v={}" \
                    .format(result.vid)
                video = pafy.new(url)
                if is_valid(video):
                    # download video
                    video_file = download(video, cfg.video_root,
                                          result.vid)
            except:
                print("--> error occurs! skipping {}".format(result.vid))
                continue
