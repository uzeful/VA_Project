"""Test script for Youtube-8M feature extractor."""

import os
import numpy as np

import init_path
import misc.config as cfg
from misc.utils import concat_feat, get_dataloader, make_cuda, make_variable
from misc.writer import RecordWriter
from models import PCAWrapper, inception_v3

def list_folders(data_dir):
    """ Search all the subfolders, each of which contains the materials of one video """
    subfolders= []
    data_dir = os.path.expanduser(data_dir)
    for root, subdirs, filenames, in sorted(os.walk(data_dir)):
        subfolders = subdirs
        break
    return [os.path.join(data_dir, subfolder) for subfolder in subfolders]


def quantize(embedding, QUANTIZE_MIN_VAL=-2, QUANTIZE_MAX_VAL=2):
    # Quantize by:
    # - clipping to [min, max] range
    clipped_embeddings = np.clip(
        embedding, QUANTIZE_MIN_VAL,
        QUANTIZE_MAX_VAL)
    # - convert to 8-bit in range [0.0, 255.0]
    quantized_embeddings = (
        (clipped_embeddings - QUANTIZE_MIN_VAL) *
        (255.0 /
         (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL)))
    # - cast 8-bit float to uint8
    quantized_embeddings = quantized_embeddings.astype(np.uint8)

    return quantized_embeddings


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

    subfolders = list_folders(cfg.dataset_path)
    for subfolder in subfolders:
        print("current folder: {}".format(subfolder))
        # data loader for frames in single video
        data_loader = get_dataloader(dataset="FrameImage",
                                     path=os.path.join(subfolder, 'frames'),
                                     frame_num =cfg.frame_num,
                                     batch_size=cfg.batch_size)

        # extract features by inception_v3
        feats = None
        for step, frames in enumerate(data_loader):
            #print("extracting feature [{}/{}]".format(step + 1, len(data_loader)))
            feat = model(make_variable(frames))
            feat_np = feat.data.cpu().numpy()
            # recude dimensions by PCA
            feat_ = pca.transform(feat_np)
            feats = concat_feat(feats, feat_)
            embedding = quantize(feats)

        # write visual features into numpy file
        np.save('{}.npy'.format(os.path.join(subfolder, 'vfeat')), embedding[:cfg.frame_num,])
