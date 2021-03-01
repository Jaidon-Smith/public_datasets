"""jsut dataset."""

import tensorflow_datasets as tfds
import os
import tensorflow as tf

# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Japanese speech corpus, named the "JSUT corpus," that is aimed at achieving end-to-end speech synthesis. The corpus consists of 10 hours of reading-style speech data and its transcription and covers all of the main pronunciations of daily-use Japanese characters.
"""

# BibTeX citation
_CITATION = """
@ARTICLE{2017arXiv171100354S,
       author = {{Sonobe}, Ryosuke and {Takamichi}, Shinnosuke and {Saruwatari}, Hiroshi},
        title = "{JSUT corpus: free large-scale Japanese speech corpus for end-to-end speech synthesis}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language},
         year = 2017,
        month = oct,
          eid = {arXiv:1711.00354},
        pages = {arXiv:1711.00354},
archivePrefix = {arXiv},
       eprint = {1711.00354},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017arXiv171100354S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


"""


class Jsut(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for jsut dataset."""

  VERSION = tfds.core.Version('1.1.0')
  RELEASE_NOTES = {
      '1.1.0': 'Current release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place the `jsut_ver1.1.zip`
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
        homepage='https://sites.google.com/site/shinnosuketakamichi/publication/jsut',
        citation=_CITATION,
        metadata=tfds.core.MetadataDict(sample_rate=48000),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(jsut): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`

    #raise Exception(dl_manager.manual_dir)
    try:
      print("The manual directory is: ", dl_manager.manual_dir)
    except:
      raise Exception("You have not defined the manual directoy where the .zip is located")
    archive_path = dl_manager.manual_dir / 'jsut_ver1.1.zip'
    #archive_path = 'hello2/jsut_ver1.1.zip'

    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"directory": extracted_path / 'jsut_ver1.1'},
        ),
    ]

  def _generate_examples(self, directory):
    """Yields examples."""
    # TODO(jsut): Yields (key, example) tuples from the dataset
    folders = [
      'basic5000',
      'utparaphrase512',
      'onomatopee300',
      'countersuffix26',
      'loanword128',
      'voiceactress100',
      'travel1000',
      'precedent130',
      'repeat500',
    ]
    for folder in folders:
      metadata_path = os.path.join(directory, folder, 'transcript_utf8.txt')
      with tf.io.gfile.GFile(metadata_path) as f:
        for line in f:
            line = line.strip()
            key, transcript = line.split(":")
            wav_path = os.path.join(directory, folder, "wav",
                                      "%s.wav" % key)
            example = {
            "id": key,
            "speech": wav_path,
            "text": transcript,
            }
            yield key, example
