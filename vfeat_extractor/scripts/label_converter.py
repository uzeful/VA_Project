"""Label converter for Youtube-8M."""

import csv

import init_path
from misc import config as cfg

vocab = None
with open(cfg.vocab_path, mode='r') as f:
    reader = csv.reader(f)
    vocab = {rows[0]: rows[3] for rows in reader}


def convert_labels(index):
    """Convert labels into names."""
    return vocab[str(index)]


def parse_pred_csv(filepath):
    """Parse predicted labels."""
    with open(filepath, mode='r') as f:
        reader = csv.reader(f)
        rows = [r for r in reader]
        vid = rows[1][0]
        preds_raw = rows[1][1].split()
        pred_names = []
        pred_probs = []
        for idx in range(len(preds_raw) // 2):
            pred_names.append(convert_labels(preds_raw[2 * idx]))
            pred_probs.append(preds_raw[2 * idx + 1])
        return vid, pred_names, pred_probs


if __name__ == '__main__':
    vid, pred_names, pred_probs = parse_pred_csv(cfg.pred_path)
    print("[vid]: {}".format(vid))
    for idx in range(cfg.top_k):
        print("  [{}]:\t{}".format(pred_names[idx], pred_probs[idx]))
