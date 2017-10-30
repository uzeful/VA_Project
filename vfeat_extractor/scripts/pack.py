"""Pack script for Youtube-8M feature extractor."""

import os

import tensorflow as tf
import torch

import init_path
import misc.config as cfg
from misc.reader import Reader
from misc.utils import concat_feat, get_dataloader, make_cuda, make_variable
from misc.writer import RecordWriter
from models import PCAWrapper

if __name__ == '__main__':
    # init PCA model
    print("=== init PCA model ===")
    pca = PCAWrapper(n_components=cfg.n_components,
                     batch_size=cfg.pca_batch_size)
    pca.load_params(filepath=cfg.pca_model)

    # get record list
    print("=== get recorf list ===")
    print("find record file in: {}".format(cfg.record_root))
    record_list = os.listdir(cfg.record_root)
    record_list = [r for r in record_list
                   if os.path.splitext(r)[1] == ".tfrecord"]
    print("total files: {}".format(len(record_list)))

    # extract features
    print("=== extract features ===")
    save_counter = 0
    feat_counter = 0
    writer = RecordWriter(
        filepath=cfg.extract_feat_path.format(save_counter),
        level="frame")
    for idx, record_file in enumerate(record_list):
        print(">>> reading record file: {} [{}/{}]".format(record_file,
                                                           idx + 1,
                                                           len(record_file)))
        for record in tf.python_io.tf_record_iterator(
                os.path.join(cfg.record_root, record_file)):
            result = Reader(record)

            if not os.path.exists(
                    cfg.inception_v3_feats_path.format(result.vid)):
                print("skipping {}".format(result.vid))
                continue
            else:
                print("extracting {} - labels {}".format(result.vid,
                                                         result.labels))
                print("--> load fatures")
                feats = torch.load(
                    cfg.inception_v3_feats_path.format(result.vid))
                print("--> recude dimensions by PCA")
                feats_ = pca.transform(feats.numpy())
                print("--> write to tfrecord")
                writer.write(vid=result.vid,
                             feat_rgb=feats_,
                             labels=result.labels)
                feat_counter += 1

                if (feat_counter + 1) % cfg.feats_per_file == 0:
                    print(">>> saving tfrecord: {}"
                          .format(cfg.extract_feat_path.format(save_counter)))
                    writer.close()
                    save_counter += 1
                    feat_counter = 0
                    writer = RecordWriter(
                        filepath=cfg.extract_feat_path.format(save_counter),
                        level="frame")
