"""tatoeba_japanese dataset."""

import tensorflow_datasets as tfds
import os
import tensorflow as tf

# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# BibTeX citation
_CITATION = """
"""


class TatoebaJapanese(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for tatoeba_japanese dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Current release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place the `tatoeba_japanese.zip`
  file in the `manual_dir/`.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "id": tf.string,
            "speech": tfds.features.Audio(sample_rate=32000),
            "text": tfds.features.Text(),
        }),
        supervised_keys=("text", "speech"),
        homepage='https://tatoeba.org/eng/downloads',
        citation=_CITATION,
        metadata=tfds.core.MetadataDict(sample_rate=32000),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`

    #raise Exception(dl_manager.manual_dir)
    try:
      print("The manual directory is: ", dl_manager.manual_dir)
    except:
      raise Exception("You have not defined the manual directoy where the .zip is located")
    archive_path = dl_manager.manual_dir / 'tatoeba_japanese.zip'

    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"directory": extracted_path / 'tatoeba_japanese'},
        ),
    ]

  def _generate_examples(self, directory):
    """Yields examples."""
    # Yields (key, example) tuples from the dataset
    metadata_path = os.path.join(directory, 'sentences.txt')
    with tf.io.gfile.GFile(metadata_path) as f:
        for line in f:
            line = line.strip()
            path, transcript = line.split(" ")
            wav_path = os.path.join(directory, "clips",
                                      path)
            key = path.split('.mp3')[0]
            example = {
            "id": key,
            "speech": wav_path,
            "text": transcript,
            }
            yield key, example
