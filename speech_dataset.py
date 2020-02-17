# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Additionally, (c) 2020 Edgar Liberis for modifications in upgrading
# to TF 2.x and the Dataset API.

"""Model definitions for simple speech recognition.
"""

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import urllib

import numpy as np
import tensorflow as tf

from tensorflow.python.util import compat

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
    from tensorflow.lite.experimental.microfrontend.python.ops import \
        audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
    frontend_op = None

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.
    Args:
        wanted_words: List of strings containing the custom words.
    Returns:
        List with the standard silence and unknown tokens added.
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""
    def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
                 wanted_words, validation_percentage, testing_percentage, model_settings):
        if data_dir:
            self.data_dir = data_dir
            self.maybe_download_and_extract_dataset(data_url, data_dir)
            self.prepare_data_index(silence_percentage, unknown_percentage,
                                    wanted_words, validation_percentage,
                                    testing_percentage)
            self.prepare_background_data()
        self.model_settings = model_settings

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        """Download and extract data set tar file.
        If the data set we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a
        directory.
        If the data_url is none, don't download anything and expect the data
        directory to contain the correct files already.
        Args:
          data_url: Web location of the tar file containing the data set.
          dest_directory: File path to extract data to.
        """
        if not data_url:
            return
        if not tf.io.gfile.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not tf.io.gfile.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                tf.compat.v1.logging.error(
                    'Failed to download URL: %s to folder: %s', data_url, filepath)
                tf.compat.v1.logging.error(
                    'Please make sure you have enough free space and'
                    ' an internet connection')
                raise
            statinfo = os.stat(filepath)
            tf.compat.v1.logging.info('Successfully downloaded %s (%d bytes)',
                                      filename, statinfo.st_size)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def _which_set(self, filename, validation_percentage, testing_percentage):
        """Determines which data partition the file should belong to.
        We want to keep files in the same training, validation, or testing sets even
        if new ones are added over time. This makes it less likely that testing
        samples will accidentally be reused in training when long runs are restarted
        for example. To keep this stability, a hash of the filename is taken and used
        to determine which set it should belong to. This determination only depends on
        the name and the set proportions, so it won't change as other files are added.
        It's also useful to associate particular files as related (for example words
        spoken by the same person), so anything after '_nohash_' in a filename is
        ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
        'bobby_nohash_1.wav' are always in the same set, for example.
        Args:
          filename: File path of the data sample.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.
        Returns:
          String, one of 'training', 'validation', or 'testing'.
        """
        base_name = os.path.basename(filename)
        # We want to ignore anything after '_nohash_' in the file name when
        # deciding which set to put a wav in, so the data set creator has a way of
        # grouping wavs that are close variations of each other.
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        # This looks a bit magical, but we need to decide whether this file should
        # go into the training, testing, or validation sets, and we want to keep
        # existing files in the same set even if more files are subsequently
        # added.
        # To do that, we need a stable way of deciding based on just the file name
        # itself, so we do a hash of that and then use that to generate a
        # probability value that we use to assign it.
        hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (MAX_NUM_WAVS_PER_CLASS + 1)) *
                           (100.0 / MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < validation_percentage:
            result = 'validation'
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = 'testing'
        else:
            result = 'training'
        return result

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        """Prepares a list of the samples organized by set and label.
        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.
        Args:
          silence_percentage: How much of the resulting data should be background.
          unknown_percentage: How much should be audio outside the wanted classes.
          wanted_words: Labels of the classes we want to be able to recognize.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.
        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.
        Raises:
          Exception: If expected files are not found.
        """
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in tf.io.gfile.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = self._which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append((wav_path, word))
            else:
                unknown_index[set_index].append((wav_path, word))
        if not all_words:
            raise Exception(f"No .wavs found at {search_path}")
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception(f"Expected to find {wanted_word} in labels "
                                f"but only found {', '.join(all_words.keys())})")
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0][0]
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append((silence_wav_path, SILENCE_LABEL))
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        """Searches a folder for background noise audio, and loads it into memory.
        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.
        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.
        Returns:0.0,
                                     0.0, 0, 'validation'
          List of raw PCM-encoded audio samples of background noise.
        Raises:
          Exception: If files aren't found in the folder.
        """
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not tf.io.gfile.exists(background_dir):
            return self.background_data
        search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
        for wav_path in tf.io.gfile.glob(search_path):
            wav_loader = tf.io.read_file(wav_path)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
            wav_data = wav_decoder.audio.numpy().flatten()
            self.background_data.append(wav_data)
        if not self.background_data:
            raise Exception('No background wav files were found in ' + search_path)

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.
        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.
        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_dataset(self, background_frequency, background_volume_range, time_shift, mode):
        candidates = self.data_index[mode]

        data = tf.data.Dataset.from_tensor_slices(candidates)

        settings = self.model_settings
        desired_samples = settings['desired_samples']
        sample_rate = settings['sample_rate']
        window_size_ms = (settings['window_size_samples'] * 1000) / sample_rate
        window_step_ms = (settings['window_stride_samples'] * 1000) / sample_rate
        num_channels = settings['fingerprint_width']
        spectrogram_length = settings['spectrogram_length']

        use_background = self.background_data and (mode == 'training')

        if mode == "training":
            data = data.shuffle(buffer_size=len(candidates) // 2)

        def load(sample):
            file, label = sample
            label = label.numpy().decode("ascii")

            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]

            # Choose a section of background noise to mix in.
            if use_background or label == SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) <= desired_samples:
                    raise ValueError(f"Background sample is too short! Need more than {desired_samples} samples but only"
                                     f" {len(background_samples)} were found")
                background_offset = np.random.randint(0, len(background_samples) - desired_samples)
                background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
                background_data = background_clipped.reshape([desired_samples, 1])

                if label == SILENCE_LABEL:
                    background_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0.0
            else:
                background_data = np.zeros([desired_samples, 1])
                background_volume = 0.0

            # If we want silence, mute out the main sample but leave the background.
            foreground_volume = 0.0 if label == SILENCE_LABEL else 1.0

            # Load and augment the data file
            wav_loader = tf.io.read_file(file)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
            # Allow the audio sample's volume to be adjusted.
            scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume)
            # Shift the sample's start position, and pad any gaps with zeros.
            padded_foreground = tf.pad(
                tensor=scaled_foreground,
                paddings=time_shift_padding,
                mode='CONSTANT')
            sliced_foreground = tf.slice(padded_foreground,
                                         time_shift_offset,
                                         [desired_samples, -1])

            # Mix in background noise.
            background_volume = tf.cast(background_volume, tf.float32)
            background_mul = tf.multiply(background_data, background_volume)
            background_add = tf.add(background_mul, sliced_foreground)
            background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
            int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)

            micro_frontend = frontend_op.audio_microfrontend(
                int16_input,
                sample_rate=sample_rate,
                window_size=window_size_ms,
                window_step=window_step_ms,
                num_channels=num_channels,
                out_scale=1,
                out_type=tf.float32)
            return tf.multiply(micro_frontend, (10.0 / 256.0)), self.word_to_index[label]

        def apply_preprocessing(i):
            x, y = tf.py_function(load, [i], [tf.float32, tf.int32])
            x.set_shape((spectrogram_length, num_channels))
            y.set_shape(())
            x = tf.expand_dims(x, axis=-1)
            return x, y

        data = data.map(apply_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return data


class SpeechDataset:
    """Wraps AudioProcessor from the original implementation into something that hides invocation complexity a bit."""
    def __init__(self, words=None,
                 data_dir="/content/speech_dataset",
                 data_url="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                 clip_duration_ms=1000,
                 feature_bin_count=40,
                 sample_rate=16000,
                 silence_percentage=25,
                 unknown_percentage=25,
                 validation_percentage=10,
                 testing_percentage=10,
                 window_size_ms=30.0,
                 window_stride=20):
        wanted_words = words if words is not None else ["yes", "no"]
        self.prepared_words_list = prepare_words_list(wanted_words)
        self.model_settings = \
            self._prepare_model_settings(len(self.prepared_words_list), sample_rate, clip_duration_ms,
                                         window_size_ms, window_stride, feature_bin_count)

        self.audio_processor = AudioProcessor(data_url, data_dir, silence_percentage, unknown_percentage, wanted_words,
                                              validation_percentage, testing_percentage, self.model_settings)

    def _prepare_model_settings(self, label_count, sample_rate, clip_duration_ms,
                                window_size_ms, window_stride_ms, feature_bin_count):
        desired_samples = int(sample_rate * clip_duration_ms / 1000)
        window_size_samples = int(sample_rate * window_size_ms / 1000)
        window_stride_samples = int(sample_rate * window_stride_ms / 1000)
        length_minus_window = (desired_samples - window_size_samples)
        if length_minus_window < 0:
            spectrogram_length = 0
        else:
            spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

        average_window_width = -1
        fingerprint_width = feature_bin_count
        fingerprint_size = fingerprint_width * spectrogram_length

        return {
            'desired_samples': desired_samples,
            'window_size_samples': window_size_samples,
            'window_stride_samples': window_stride_samples,
            'spectrogram_length': spectrogram_length,
            'fingerprint_width': fingerprint_width,
            'fingerprint_size': fingerprint_size,
            'label_count': label_count,
            'sample_rate': sample_rate,
            'average_window_width': average_window_width,
        }

    def training_dataset(self, time_shift_ms=100.0, background_frequency=0.8, background_volume=0.1):
        time_shift_samples = int((time_shift_ms * self.model_settings['sample_rate']) / 1000)
        return self.audio_processor.get_dataset(background_frequency, background_volume,
                                                time_shift_samples, 'training')

    def validation_dataset(self, time_shift_ms=0.0, background_frequency=0.0, background_volume=0.0):
        time_shift_samples = int((time_shift_ms * self.model_settings['sample_rate']) / 1000)
        return self.audio_processor.get_dataset(background_frequency, background_volume,
                                                time_shift_samples, 'validation')

    def testing_dataset(self, time_shift_ms=0.0, background_frequency=0.0, background_volume=0.0):
        time_shift_samples = int((time_shift_ms * self.model_settings['sample_rate']) / 1000)
        return self.audio_processor.get_dataset(background_frequency, background_volume,
                                                time_shift_samples, 'validation')

    def sample_shape(self):
        input_frequency_size = self.model_settings['fingerprint_width']
        input_time_size = self.model_settings['spectrogram_length']
        return (input_time_size, input_frequency_size, 1)

    def label_count(self):
        return self.model_settings['label_count']

    def look_up_word(self, index):
        return self.prepared_words_list[index]


if __name__ == "__main__":
    pass
