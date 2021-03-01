# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
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

"""Mozilla Common Voice Dataset."""

import collections
import csv
import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DOWNLOAD_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/{}.tar.gz"
_SPLITS = {
    tfds.Split.TRAIN: "train",
    tfds.Split.TEST: "test",
    tfds.Split.VALIDATION: "validated"
}
_GENDER_CLASSES = ["male", "female", "other"]

# OrderedDict to keep collection order constant (en is the default config)
_LANGUAGES = ["en", "de", "fr", "cy", "br", "cv", "tr", "tt", "ky", "ga-IE", "kab", "ca", "zh-TW", "sl", "it", "nl", "cnh", "eo", "ja"]


class CommonVoiceConfig(tfds.core.BuilderConfig):
  """Configuration Class for Mozilla CommonVoice Dataset."""
  
  def __init__(self, *, language, **kwargs):
    """Constructs CommonVoiceConfig.
    Args:
     language: `str`, one of [ca, nl, br, de, sl, cy, en, kab, tt, zh-TW, eo,
       it, fr, ga-IE, tr, ky, cnh, cv]. Language Code of the Dataset to be used.
     **kwargs: keywords arguments forwarded to super
    """
    if language not in _LANGUAGES:
      raise ValueError("language must be one of {}. Not: {}".format(
          _LANGUAGES, language))
    self.language = language

    kwargs.setdefault("name", language)
    kwargs.setdefault("description", "Language Code: %s" % language)
    kwargs.setdefault("version", tfds.core.Version("1.0.0"))
    super(CommonVoiceConfig, self).__init__(**kwargs)


class CommonVoiceV6(tfds.core.GeneratorBasedBuilder):

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place the `(language).tar`
  file in the `manual_dir/`.
  """

  """Mozilla Common Voice Dataset."""
  BUILDER_CONFIGS = [
      CommonVoiceConfig(language=l)
      for l in _LANGUAGES
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        description=("Mozilla Common Voice Dataset"),
        builder=self,
        features=tfds.features.FeaturesDict({
            "client_id":
                tfds.features.Text(),
            "upvotes":
                tf.int32,
            "downvotes":
                tf.int32,
            "age":
                tfds.features.Text(),
            "gender":
                tfds.features.ClassLabel(names=_GENDER_CLASSES),
            "sentence":
                tfds.features.Text(),
            "voice":
                tfds.features.Audio(),
        }),
        homepage="https://voice.mozilla.org/en/datasets",
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    #extracted_path = dl_manager.download_and_extract(
    #    _DOWNLOAD_URL.format(self.builder_config.language))
    try:
      print("The manual directory is: ", dl_manager.manual_dir)
    except:
      raise Exception("You have not defined the manual directoy where the .zip is located")


    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)


    clip_folder = os.path.join(extracted_path, "clips")
    return [
        tfds.core.SplitGenerator(  # pylint: disable=g-complex-comprehension
            name=k,
            gen_kwargs={
                "audio_path": clip_folder,
                "label_path": os.path.join(extracted_path, "%s.tsv" % v)
            },
        ) for k, v in _SPLITS.items()
    ]

  def _generate_examples(self, audio_path, label_path):
    """Generate Voice samples and statements given the data paths.
    Args:
      audio_path: str, path to audio storage folder
      label_path: str, path to the label files
    Yields:
      example: The example `dict`
    """
    with tf.io.gfile.GFile(label_path) as file_:
      dataset = csv.DictReader(file_, delimiter="\t")
      for i, row in enumerate(dataset):
        file_path = os.path.join(audio_path, "%s.mp3" % row["path"])
        if tf.io.gfile.exists(file_path):
          yield i, {
              "client_id": row["client_id"],
              "voice": file_path,
              "sentence": row["sentence"],
              "upvotes": int(row["up_votes"]) if row["up_votes"] else 0,
              "downvotes": int(row["down_votes"]) if row["down_votes"] else 0,
              "age": row["age"],
              "gender": row["gender"] if row["gender"] else -1
          }
