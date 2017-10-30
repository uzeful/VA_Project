"""Check if TFRecord is valid and complete."""

import argparse
import os

import tensorflow as tf

import init_path
from misc.reader import Reader


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Read TFRecord and check its validation")
    parser.add_argument('filepath',
                        help='path for TFRecord file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    for idx, record_file in enumerate(os.listdir(args.filepath)):
        print("checking record file: {} [{}/{}]".format(
            record_file,
            idx + 1,
            len(os.listdir(args.filepath))))
        record_iterator = tf.python_io.tf_record_iterator(
            os.path.join(args.filepath, record_file))
        for record in record_iterator:
            result = Reader(record)
            # print("--> checking {}".format(result.vid))
    print("All checked!")
