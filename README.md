# cyberbully-classifier

This repository consists of multiple machine learning models that can be used to predict if online comments are classified as cyberbullying.

### DATA

`./data/get_data_subset.py` takes the `train.csv` and extracts about 2000 points (cause the original data set is really really big).
It saves the subset of data in `train_subset.csv`.

`./data/preprocess.py` cleans up the data (removing urls, punctuation, etc). Saves new data as `train_subset_clean.csv`.

All models should use `train_subset_clean.csv` for their training/testing using Cross Validation.
