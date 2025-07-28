# Model using TF-IDF and Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


CSV_PATH = "../data/output/clean_data.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_LABEL = "cyberbullying"  # Stratify to ensure the train/test splits have the same relative frequency of toxic vs. non‑toxic

# Load & read column names
df = pd.read_csv(CSV_PATH)
X = df["comment_text"].astype(str)
Y = df["cyberbullying"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
)

# Use TF-IDF to turn text into numbers + try a logistic regression model
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=None,
                stop_words=None,
                max_features=20000,
                ngram_range=(1, 2),
            ),
        ),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
        ),
    ]
)

# Parameters
params = {
    "tfidf__max_df": [
        0.9,
        0.95,
        1.0,
    ],  # Drop terms that appear in more than this fraction of documents (too common)
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
    ],  # Whether to use just unigrams (1,1) or unigrams + bigrams (1,2)
    "clf__C": [
        0.1,
        1.0,
        10.0,
    ],  # Amount of regularizatioon - bigger C = less regularization
}
grid = GridSearchCV(
    estimator=pipeline,  # TF-IDF + logistic regression pipeline
    param_grid=params,
    cv=3,  # Use 3 fold cross validation
    scoring="f1_macro",  # Trying to maximize f1
    n_jobs=-1,  # Use all CPU cores
    verbose=2,  # print progress
)

# Train and evaluate model on every combo in param_grid suing 3-fold CV, keep the best f1
grid.fit(X_train, y_train)

print("\nBest parameters: ")
for param, val in grid.best_params_.items():
    print(f"{param}: {val}")

# Evaluation
print("\nEvaluating on test set: ")
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

label = "cyberbullying"
print(f"\n--- {label.upper()} ---")
print(classification_report(y_test, y_pred, zero_division=0))

# for i, label in enumerate(Y.columns):
#     print(f"\n--- {label.upper()} ---")
#     print(classification_report(y_test[label], y_pred[:, i], zero_division=0))

# Confusion matrix
# labels = y_test.columns.tolist()  # ['cyberbullying']
# for i, lbl in enumerate(labels):
#     y_true = y_test[lbl].values
#     y_predicted = y_pred[:, i]

#     cm = confusion_matrix(y_true, y_predicted, labels=[0, 1])
#     print(f"\n--- {lbl.upper()} ---")
#     print(pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))

#     # Plot
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
#     disp.plot(values_format="d")
#     plt.title(lbl.upper())
#     plt.show()

"""
              precision    recall  f1-score   support

           0       0.98      0.95      0.97     28670
           1       0.67      0.85      0.75      3245

    accuracy                           0.94     31915
   macro avg       0.82      0.90      0.86     31915
weighted avg       0.95      0.94      0.94     31915

TO IMPROVE:
- get more data for threat and discrimination
    - oversample these classes?
- Try a different linear model --> LinearSVC does slightly better on text
"""
