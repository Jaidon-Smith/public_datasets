Various public datasets compatible with TensorFlow Datasets
# Dataset Information
## JSUT
[More information available at dataset's webpage](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)

This corpus consists of Japanese text (transcription) and reading-style audio. The audio data is sampled at 48kHz and recorded in our anechoic room. we recorded voices of a native Japanese female speaker. This corpus contains 10-hour speech consisting of the following data: 

* basic5000 ... covers all of daily-use characters (jouyou kanji).
* utparaphrase512 ... replaces a part of a sentence with its paraphrase.
* onomatopee300 ... includes onomatopees (onomatopia) of Japanese.
* countersuffix26 ... countersuffix of Japanese
* loanword128 ... loanwords of Japanese (e.g., ググる ['google' as verb])
* voiceactress100 ... para-speech to the Voice Actress Corpus (free corpus of professional female speakers)
* travel1000 ... travel-domain corpus
* precedent130 ... precedent sentences
* repeat500 ... repeatedly spoken utterances (100 sentence * 5 times)

## Common Voice Version 6
Multilingual speech corpus.
The Japanese version contains 5h of speech. Only 3h is validated.
[More information available at dataset's webpage](https://commonvoice.mozilla.org/en/datasets)
# How to Use
Make sure that [TensorFlow Datasets](https://www.tensorflow.org/datasets) Nightly is installed.
```
pip install tfds-nightly
```
Then git clone this repository with `git clone https://github.com/Jaidon-Smith/public-datasets.git`

## JSUT
Download the dataset zip file from the dataset's webpage:
[https://sites.google.com/site/shinnosuketakamichi/publication/jsut](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
```python
import jsut
```
Install pydub
```
pip install pydub
```

Load the dataset, it is necessary to specify the manual directory where the zip file was placed.
data_dir argument is optional.

```python
download_config = tfds.download.DownloadConfig(manual_dir='/gdrive/MyDrive/datasets/public_datasets/downloads/manual')

ds = tfds.load("jsut", data_dir='/gdrive/MyDrive/datasets/public_datasets', download_and_prepare_kwargs={"download_config": download_config})
```

## Common Voice Version 6
Download the dataset tar file for the desired language from the dataset's webpage:
[https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)
```python
import common_voice_v6
```
Install pydub
```
pip install pydub
```
Load the dataset, it is necessary to specify the manual directory where the tar file was placed.
data_dir argument is optional.
```
download_config = tfds.download.DownloadConfig(manual_dir='/gdrive/MyDrive/datasets/public_datasets/downloads/manual/commonvoice')

ds = tfds.load("common_voice_v6/ja", data_dir='/gdrive/MyDrive/datasets/public_datasets', download_and_prepare_kwargs={"download_config": download_config})
```

