"""Test script for Youtube-8M feature extractor."""

import os
import numpy as np

import init_path
import misc.config as cfg
from misc.utils import concat_feat, get_dataloader, make_cuda, make_variable
from misc.writer import RecordWriter
from models import PCAWrapper, inception_v3

if __name__ == '__main__':
    # init Inception v3 model
    model = make_cuda(inception_v3(pretrained=True,
                                   model_path=cfg.inception_v3_model,
                                   transform_input=True,
                                   extract_feat=True))
    model.eval()

    # init PCA model
    pca = PCAWrapper(n_components=cfg.n_components,
                     batch_size=cfg.pca_batch_size)
    pca.load_params(filepath=cfg.pca_model)

    # data loader for frames in ingle video
    data_loader = get_dataloader(dataset="FrameImage",
                                 path=os.path.join(cfg.frame_path, 'frames'),
                                 frame_num =cfg.frame_num,
                                 batch_size=cfg.batch_size)

    # extract features by inception_v3
    feats = None
    for step, frames in enumerate(data_loader):
        print("extracting feature [{}/{}]".format(step + 1, len(data_loader)))
        feat = model(make_variable(frames))
        feat_np = feat.data.cpu().numpy()
        # recude dimensions by PCA
        feat_ = pca.transform(feat_np)
        feats = concat_feat(feats, feat_)

    # write features into TFRecord
    np.save('{}.npy'.format(os.path.join(cfg.frame_path, 'video_feature')), feats)
