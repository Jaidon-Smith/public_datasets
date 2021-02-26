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

Load the dataset, it is necessary to specify the manual directory where the zip file should be placed.
data_dir argument is optional.

