# public-datasets
Various public datasets compatible with TensorFlow Datasets

# How to Use
Make sure that [TensorFlow Datasets](https://www.tensorflow.org/datasets) is installed.
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
download_config = tfds.download.DownloadConfig(manual_dir='/gdrive/MyDrive/datasets/tensorflow_datasets/downloads/manual')

ds = tfds.load("jsut", data_dir='/gdrive/MyDrive/datasets/tensorflow_datasets', download_and_prepare_kwargs={"download_config": download_config})
```

