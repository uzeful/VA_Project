# yt8m-feature-extractor
Extract features from video file as the format in Youtube-8M.

## Description

- `scripts/download.py`: download videos from YouTube corresponding to the TFRecord file.
- `scripts/decode.py`: decode frames from video and save them to data folder.
- `scripts/train_pca.py`: load extracted inception_v3 features and fit PCA with them.
- `scripts/test.py`: test single video file and generate TFRecord.
- `scripts/extract.py`: extract inception_v3 features from decoded image folders.
- `scripts/pack.py`: transform and pack your downloaded videos into Youtube-8M-dataset-like TFRecord file.
- `scripts/pipeline.py`: download videos and extract inception_v3 features.
- `scripts/label_converter.py`: convert label numbers into names.
- `scripts/checker.py`: check if downloaded TFRecord is valid and complete.
- `demo.sh`: all-in-one shell script for testing single video file and get its tags.

## Workflow

1. Run `virtualenv -p python3 yt8m-env && source yt8m-env/bin/activate` for virtual Python environment.
2. Run `pip3 install -r requirements.txt` for required Python packages.
  - You may install `pytorch` by the instruction of [its official website](pytorch.org).
3. Modify `misc/config.py` for custom configuration.
4. Run `python3 scripts/pipeline.py` to download videos and extract inception_v3 features.
5. Once you've downloaded enough videos, you can run `python3 scripts/train_pca.py` to fit pca.
6. After fitting PCA, run `python3 scripts/pack.py` to transform and pack your downloaded videos into Youtube-8M-dataset-like TFRecord file.
7. Just run your training scripts for Youtube-8M and enjoy!
