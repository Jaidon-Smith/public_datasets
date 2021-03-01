"""jvs dataset."""

import tensorflow_datasets as tfds
import os
import tensorflow as tf

# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """"""

# BibTeX citation
_CITATION = """
"""


class Jvs(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for jvs dataset."""

  VERSION = tfds.core.Version('1.1.0')
  RELEASE_NOTES = {
      '1.1.0': 'Current release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place the `jvs_ver1.zip`
  file in the `manual_dir/`.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "id": tf.string,
            "speech": tfds.features.Audio(sample_rate=48000),
            "text": tfds.features.Text(),
        }),
        supervised_keys=("text", "speech"),
        homepage='https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus',
        citation=_CITATION,
        metadata=tfds.core.MetadataDict(sample_rate=48000),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`

    #raise Exception(dl_manager.manual_dir)
    try:
      print("The manual directory is: ", dl_manager.manual_dir)
    except:
      raise Exception("You have not defined the manual directoy where the .zip is located")
    archive_path = dl_manager.manual_dir / 'jvs_ver1.zip'

    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"directory": extracted_path / 'jvs_ver1'},
        ),
    ]

  def _generate_examples(self, directory):
    """Yields examples."""
    folders = tf.io.gfile.glob(os.path.join(directory, "*", ""))
    folders = [i for i in os.listdir(directory) if os.path.isdir(os.path.join(directory, i))]
    for folder in folders:
      print("Folder:", folder)
      metadata_path = os.path.join(directory, folder, 'transcripts_utf8.txt')
      with tf.io.gfile.GFile(metadata_path) as f:
        for line in f:
            line = line.strip()
            key, transcript = line.split(":")
            wav_path = os.path.join(directory, folder, "wav24kHz16bit",
                                      "%s.wav" % key)
            id = str(folder).split('/')[-2] + key
            example = {
            "id": id,
            "speech": wav_path,
            "text": transcript,
            }
            yield id, example
