"""Reader for Youtube-8M dataset."""

import numpy as np
import tensorflow as tf

from misc.utils import dequantize, resize_axis


class Reader(object):
    """Reader for Youtube-8M dataset."""

    def __init__(self, record):
        """Init Reader."""
        super(Reader, self).__init__()
        self.vid = None
        self.labels = None
        self.feat_rgb = None
        self.feat_audio = None

        self.load(record)

    def load(self, record):
        """Load TFRecord."""
        result = tf.train.Example.FromString(record)
        feature = result.features.feature
        self.vid = feature["video_id"].bytes_list.value[0].decode("utf-8")
        self.labels = list(feature["labels"].int64_list.value)
        self.feat_rgb = np.array(list(feature["mean_rgb"].float_list.value))
        self.feat_audio = np.array(
            list(feature["mean_audio"].float_list.value))


class RecordReader(object):
    """RecordReader for Youtube-8M dataset."""

    def __init__(self, filepath, level="frame", num_frames=300):
        """Init RecordReader."""
        super(RecordReader, self).__init__()
        assert level == "frame" or level == "frame", \
            "yt8m-level must be `frame` or `video`"
        self.filepath = filepath
        self.level = level
        self.num_frames = num_frames
        self.reader = tf.TFRecordReader()

    def read(self):
        """Read."""
        filename_queue = tf.train.string_input_producer([self.filepath])
        _, serialized_example = self.reader.read(filename_queue)
        if self.level == "frame":
            features, sequence_features = tf.parse_single_sequence_example(
                serialized_example,
                context_features={
                    "video_id": tf.FixedLenFeature([], tf.string),
                    "labels": tf.VarLenFeature(tf.int64),
                },
                sequence_features={
                    "rgb": tf.FixedLenSequenceFeature([], tf.string),
                    "audio": tf.FixedLenSequenceFeature([], tf.string),
                }
            )
        elif self.level == "video":
            features, sequence_features = tf.parse_single_example(
                serialized_example,
                features={
                    "video_id": tf.FixedLenFeature([], tf.string),
                    "labels": tf.VarLenFeature(tf.int64),
                    "mean_rgb": tf.FixedLenFeature([], tf.string),
                    "mean_audio": tf.FixedLenFeature([], tf.string),
                }
            )

    def decode(self,
               features,
               feature_size,
               max_frames,
               max_quantized_value=2,
               min_quantized_value=-2):
        """Decode features from an input string and dequantizes it."""
        decoded_features = tf.reshape(
            tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
            [-1, feature_size])

        num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
        feature_matrix = dequantize(decoded_features,
                                    max_quantized_value,
                                    min_quantized_value)
        feature_matrix = resize_axis(feature_matrix, 0, max_frames)
        return feature_matrix, num_frames
