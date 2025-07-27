# cyberbully-classifier

This is a machine learning model that leverages linear regression to predict cyberbullying text messages.

### DATA

`./data/get_data_subset.py` takes the `train.csv` and extracts about 2000 points (cause the original data set is really really big).
It saves the subset of data in `train_subset.csv`.

`./logisitic_regression_model/preprocess.py` cleans up the data (removing urls, punctuation, etc). Saves new data as `train_subset_clean.csv`.

Can use `train_subset_clean.csv` for models.