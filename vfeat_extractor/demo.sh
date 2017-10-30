#!/bin/bash
# supress tensorflow warnings
export TF_CPP_MIN_LOG_LEVEL=3
# generate tfrecord
source ./yt8m-env/bin/activate
python3 scripts/test.py
# inference
cd ../Youtube-8M-WILLOW
source ./willow-env/bin/activate
python inference.py --output_file="/media/m/E/download/csv/pred.csv" --input_data_pattern="/home/m/workspace/yt8m-feature-extractor/data/test.tfrecord" --model=GruModel --train_dir=GRU-0002-1200 --frame_features=True --feature_names="rgb" --feature_sizes="1024" --batch_size=1 --base_learning_rate=0.0002 --gru_cells=1200 --learning_rate_decay=0.9 --moe_l2=1e-6 --run_once=True --top_k=50
# convert labels
cd ../yt8m-feature-extractor/
source ./yt8m-env/bin/activate
python3 scripts/label_converter.py
