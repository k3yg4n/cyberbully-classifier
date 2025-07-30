# cyberbully-classifier

This repository consists of multiple machine learning models that can be used to predict if online comments are classified as cyberbullying.

## Original Dataset

The initial dataset before processing can be found in data/input/`original_data.csv`.

Sourced from `train.csv` from https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

## Preprocessing

By running `data/get_dataset.ipynb`, one can generate the dataset used to train/test for each model, at `data/output/tfidf_dataset.csv`.

## Models

Each model is implemented in a Jupyter notebook located under an appropriately labelled directory. For example, `svm/svm.ipynb`. Each notebook implements the model and outputs performance metrics for the results. See each file for additional details.

## Presentation/Report

The presentation for the project can be found at `presentation.pdf`.
