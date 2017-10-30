"""Train PCA for Youtube-8M feature extractor."""

import os

import torch

import init_path
import misc.config as cfg
from misc.utils import concat_feat_var
from models import PCAWrapper

if __name__ == '__main__':
    # init models and data loader
    pca = PCAWrapper(n_components=cfg.n_components,
                     batch_size=cfg.pca_batch_size)

    if os.path.exists(cfg.inception_v3_feats_path.format("total")):
        inception_v3_feats = torch.load(
            cfg.inception_v3_feats_path.format("total"))

    else:
        # get inception_v3 feats list
        feats_list = os.listdir(cfg.inception_v3_feats_root)
        feats_list = [v for v in feats_list
                      if os.path.splitext(v)[1] in cfg.inception_v3_feats_ext]

        # load inception_v3 feats
        inception_v3_feats = None
        for step, feat_file in enumerate(feats_list):
            print("loadingg inception_v3 from {} [{}/{}]"
                  .format(feat_file, step + 1, len(feats_list)))
            feat_path = os.path.join(cfg.inception_v3_feats_root, feat_file)
            feat = torch.load(feat_path)
            inception_v3_feats = concat_feat_var(inception_v3_feats, feat)

        # save all feats into single file
        torch.save(inception_v3_feats,
                   cfg.inception_v3_feats_path.format("total"))

    # train PCA
    X = inception_v3_feats.numpy()
    pca.fit(X)

    # sabe PCA params
    pca.save_params(filepath=cfg.pca_model)
