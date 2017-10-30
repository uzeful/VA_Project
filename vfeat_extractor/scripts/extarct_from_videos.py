"""Extract inception_v3_feats from videos for Youtube-8M feature extractor."""

import os

import torch

import init_path
import misc.config as cfg
from misc.utils import (concat_feat_var, get_dataloader, make_cuda,
                        make_variable)
from models import inception_v3

if __name__ == '__main__':
    # init models and data loader
    model = make_cuda(inception_v3(pretrained=True,
                                   transform_input=True,
                                   extract_feat=True))
    model.eval()

    # get vid list
    video_list = os.listdir(cfg.video_root)
    video_list = [v for v in video_list
                  if os.path.splitext(v)[1] in cfg.video_ext]

    # extract features by inception_v3
    for idx, video_file in enumerate(video_list):
        vid = os.path.splitext(video_file)[0]
        filepath = os.path.join(cfg.video_root, video_file)
        if os.path.exists(cfg.inception_v3_feats_path.format(vid)):
            print("skip {}".format(vid))
        else:
            print("processing {}".format(vid))
            # data loader for frames in single video
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
            # print("--> delete original video file: {}".format(filepath))
            # os.remove(filepath)
