"""Main script for Youtube-8M feature extractor."""

import misc.config as cfg
from misc.utils import concat_feat, get_dataloader, make_cuda, make_variable
from models import PCAWrapper, inception_v3

if __name__ == '__main__':
    # init models and data loader
    model = make_cuda(inception_v3(pretrained=True,
                                   transform_input=True,
                                   extract_feat=True))
    pca = PCAWrapper(n_components=cfg.n_components)
    model.eval()

    # data loader for frames in ingle video
    # data_loader = get_dataloader(dataset="VideoFrame",
    #                              path=cfg.video_file,
    #                              num_frames=cfg.num_frames,
    #                              batch_size=cfg.batch_size)
    # data loader for frames decoded from several videos
    data_loader = get_dataloader(dataset="FrameImage",
                                 path=cfg.frame_root,
                                 batch_size=cfg.batch_size)

    # extract features by inception_v3
    feats = None
    for step, frames in enumerate(data_loader):
        print("extracting feature [{}/{}]".format(step + 1, len(data_loader)))
        feat = model(make_variable(frames))
        feats = concat_feat(feats, feat.data.cpu())

    # recude dimensions by PCA
    X = feats.numpy()
    pca.fit(X)
    X_ = pca.transform(X)
    print("reduce X {} to X_ {}".format(X.shape, X_.shape))

    # sabe PCA params
    pca.save_params(filepath=cfg.pca_model)
