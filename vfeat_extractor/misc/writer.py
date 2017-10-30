"""Writer for Youtube-8M dataset."""

import tensorflow as tf

from misc.utils import encode


class RecordWriter(object):
    """RecordWriter for Youtube-8M dataset."""

    def __init__(self, filepath, level="frame"):
        """Init RecordWriter."""
        super(RecordWriter, self).__init__()
        assert level == "frame" or level == "frame", \
            "yt8m-level must be `frame` or `video`"

        self.filepath = filepath
        self.level = level
        self.writer = tf.python_io.TFRecordWriter(self.filepath)
        self.example = None

    def write(self, vid, feat_rgb, feat_audio=None, labels=None):
        """Write features into TFRecord."""
        if self.level == "frame":
            # quantize features
            feat_rgb_encoded = encode(feat_rgb)
            # feat_audio = [encode(f) for f in feat_audio]

            # Non-serial data uses Feature
            if labels is None:
                context = tf.train.Features(feature={
                    "video_id": self._bytes_feature(vid.encode()),
                })
            else:
                context = tf.train.Features(feature={
                    "video_id": self._bytes_feature(vid.encode()),
                    "labels": self._int64_list_feature(labels)
                })
            # Serial data uses FeatureLists
            feature_lists = tf.train.FeatureLists(feature_list={
                "rgb": self._bytes_feature_list(feat_rgb_encoded),
                # "audio": self._bytes_feature_list(feat_audio)
            })
            self.example = tf.train.SequenceExample(
                context=context, feature_lists=feature_lists)
        elif self.level == "video":
            #  Non-serial data uses Feature
            features = tf.train.Features(feature={
                "video_id": self._bytes_feature(vid.encode()),
                # "labels": self._int64_feature(labels),
                "mean_rgb": self._float_feature(feat_rgb),
                # "mean_audio": self._float_feature(feat_audio),
            })
            self.example = tf.train.Example(features=features)

        # Serialize To String
        if self.example is not None:
            self.writer.write(self.example.SerializeToString())

    def close(self):
        """Save TFRecord."""
        self.writer.close()

    def _int64_feature(self, value):
        """Wrapper for inserting an int64 Feature into a SequenceExample proto.

        e.g, An integer label.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_list_feature(self, values):
        """Wrapper for inserting an int64 list Feature into a SequenceExample proto.

        e.g, An list of integer labels.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _float_feature(self, value):
        """Wrapper for inserting a float Feature into a SequenceExample proto.

        e.g, An float label.
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto.

        e.g, rgb or audio feature (1024 8bit quantized features)
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature_list(self, values):
        """Wrapper for inserting an int64 FeatureList into a SequenceExample proto.

        e.g, sentence in list of ints
        """
        return tf.train.FeatureList(
            feature=[self._int64_feature(v) for v in values])

    def _float_feature_list(self, values):
        """Wrapper for inserting a float FeatureList into a SequenceExample proto.

        e.g, sentence in list of floats
        """
        return tf.train.FeatureList(
            feature=[self._float_feature(v) for v in values])

    def _bytes_feature_list(self, values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto.

        e.g, rgb or audio features
        """
        return tf.train.FeatureList(
            feature=[self._bytes_feature(v) for v in values])
