# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import os
import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import pdb

flags = tf.app.flags

flags.DEFINE_string(
    'dataset_path', '/data/datasets/yt8m',
    'dataset_path, each subfolder of which contains the wav files.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

FLAGS = flags.FLAGS


def list_folders(data_dir):
    """ Search all the subfolders, each of which contains the materials of one video """
    subfolders= []
    data_dir = os.path.expanduser(data_dir)
    for root, subdirs, filenames, in sorted(os.walk(data_dir)):
        subfolders = subdirs
        break
    return [os.path.join(data_dir, subfolder) for subfolder in subfolders]


def main(_):
    if FLAGS.dataset_path:
        subfolders = list_folders(FLAGS.dataset_path)
    #wav_files = [os.path.join(folder, '{}.wav'.format(folder.split('/')[-1])) for folder in folders]

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

        for subfolder in subfolders:
            #pdb.set_trace()
            # format the wave file name
            wav_file = os.path.join(subfolder, '{}.wav'.format(subfolder.split('/')[-1]))
            if not os.path.exists(wav_file):
                print('Skipping {}!'.format(wav_file))
                os.removedirs(subfolder)
                print('Remove dir {}!'.format(subfolder))
                continue

            print('Processing {}!'.format(wav_file))
            # transform wav_file
            examples_batch = vggish_input.wavfile_to_examples(wav_file)
            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})
            feats = pproc.postprocess(embedding_batch)
            #print(feats.shape)
            #pdb.set_trace()

            # write audio features into numpy file
            np.save('{}.npy'.format(os.path.join(subfolder, 'afeat')), feats[:120,])
        print('Audio feature extraction is finished!')

if __name__ == '__main__':
    tf.app.run()
