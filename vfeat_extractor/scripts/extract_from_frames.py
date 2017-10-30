"""Extract inception_v3_feats from images for Youtube-8M feature extractor."""

import os

import torch

import init_path
import misc.config as cfg
from misc.utils import (concat_feat_var, get_dataloader, make_cuda,
                        make_variable)
from models import PCAWrapper, inception_v3

if __name__ == '__main__':
    # init models and data loader
    model = make_cuda(inception_v3(pretrained=True,
                                   transform_input=True,
                                   extract_feat=True))
    model.eval()

    # get vid list
    video_list = os.listdir(cfg.video_root)
    video_list = [os.path.splitext(v)[0] for v in video_list
                  if os.path.splitext(v)[1] in cfg.video_ext]

    # extract features by inception_v3
    for idx, vid in enumerate(video_list):
        if os.path.exists(cfg.inception_v3_feats_path.format(vid)):
            print("skip {}".format(vid))
        else:
            print("extract feature from {} [{}/{}]".format(vid, idx + 1,
                                                           len(video_list)))
            # data loader for frames decoded from several videos
            data_loader = get_dataloader(dataset="FrameImage",
                                         path=cfg.frame_root,
                                         batch_size=cfg.batch_size,
                                         vid=vid)
            feats = None
            for step, frames in enumerate(data_loader):
                print("--> step [{}/{}]".format(step + 1,
                                                len(data_loader)))
                feat = model(make_variable(frames))
                feats = concat_feat_var(feats, feat.data.cpu())

            torch.save(feats, cfg.inception_v3_feats_path.format(vid))
