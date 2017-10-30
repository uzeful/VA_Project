"""Decode frames from video."""
import init_path
import argparse
import os
import subprocess

from moviepy.editor import *
import misc.config as cfg

import pdb


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Decode frames from video.")
    parser.add_argument("-f", '--filepath', default = '',
                        help='path for video folder')
    parser.add_argument("-s", '--start', type=int, default = 5,
            help='the start time to extract video frame')
    parser.add_argument("-e", '--end', type=int, default = 130,
            help='the end time to extract video frame')
    args = parser.parse_args()

    return args


def read(video_path, start=0, end=120):
    """
    video_path: the video path for reading
    start: the start time of the clipped video
    end: the end time of the clipped video
    """
    video = VideoFileClip(video_path)
    return video.subclip(start, end)


def decode(video=None, save_path='.', decode_rate=1):
    """
    video: video file read by moviepy.editor.VideoFileClip
    decode_rate: the frames decoded per second
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    video.write_images_sequence(nameformat=os.path.join(save_path, '%03d.jpg'), fps=1, progress_bar=True, withmask=False)


if __name__ == '__main__':
    args = parse()

    file_list = os.listdir(args.filepath)
    for step, video_file in enumerate(file_list):
        if os.path.splitext(video_file)[1] in cfg.video_ext:
            print("=== Processing {} [{}/{}]".format(video_file,
                                                     step + 1,
                                                     len(file_list)))
            try:
                video_path = os.path.join(args.filepath, video_file)
                video = read(video_path, args.start, args.end)

                save_path = os.path.join(cfg.dataset_path, video_file.split('.')[0])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                #else:
                #    continue

                decode(video, save_path=os.path.join(save_path, cfg.frame_subdir))
                prefix = os.path.join(save_path, video_file.split('.')[0])
                # extract silent video and audio files individually
                video.write_videofile(prefix+'.mp4', audio=False)
                if video.audio is not None:
                    video.audio.write_audiofile(prefix+'.wav')
            #except ValueError:
            except (ValueError, IOError, OSError):
                os.remove(video_path)
                print("skip truncated video file: {}".format(video_file))
                continue
