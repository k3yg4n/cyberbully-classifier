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


CSV_PATH = '../data/train_subset_clean.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_LABEL = 'toxic' # Stratify to ensure the train/test splits have the same relative frequency of toxic vs. non‑toxic

# Load & read column names
df = pd.read_csv(CSV_PATH)
X = df['clean_comment'].astype(str)
Y = df[['toxic', 'insult', 'threat', 'discrimination']].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=Y[STRATIFY_LABEL]
)

# Use TF-IDF to turn text into numbers + try a logistic regression model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=None,
        stop_words=None,
        max_features=20000,
        ngram_range=(1,2)
    )),
    ("clf", MultiOutputClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    ))
])

# Parameters
params = {
    "tfidf__max_df":        [0.9, 0.95, 1.0], # Drop terms that appear in more than this fraction of documents (too common)
    "tfidf__ngram_range":   [(1,1), (1,2)], # Whether to use just unigrams (1,1) or unigrams + bigrams (1,2)
    "clf__estimator__C":     [0.1, 1.0, 10.0] # Amount of regularizatioon - bigger C = less regularization
}
grid = GridSearchCV(
    estimator = pipeline, # TF-IDF + logistic regression pipeline
    param_grid = params,
    cv         = 3, # Use 3 fold cross validation
    scoring    = "f1_macro", # Trying to maximize f1
    n_jobs     = -1, # Use all CPU cores
    verbose    = 2 # print progress
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

for i, label in enumerate(Y.columns):
    print(f"\n--- {label.upper()} ---")
    print(classification_report(
        y_test[label],
        y_pred[:, i],
        zero_division=0
    ))

# Confusion matrix
labels = y_test.columns.tolist() # ['toxic','insult','threat','discrimination']

for i, lbl in enumerate(labels):
    y_true = y_test[lbl].values
    y_predicted = y_pred[:, i]

    cm = confusion_matrix(y_true, y_predicted, labels=[0,1])
    print(f"\n--- {lbl.upper()} ---")
    print(pd.DataFrame(cm, index=['true_0','true_1'], columns=['pred_0','pred_1']))

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.title(lbl.upper())
    plt.show()

'''
Evaluating on test set: 

--- TOXIC ---
              precision    recall  f1-score   support

           0       0.59      0.71      0.64       110
           1       0.88      0.81      0.85       290

    accuracy                           0.79       400
   macro avg       0.74      0.76      0.75       400
weighted avg       0.80      0.79      0.79       400


--- INSULT ---
              precision    recall  f1-score   support

           0       0.79      0.82      0.81       237
           1       0.73      0.68      0.70       163

    accuracy                           0.77       400
   macro avg       0.76      0.75      0.75       400
weighted avg       0.76      0.77      0.76       400


--- THREAT ---
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       390
           1       0.55      0.60      0.57        10

    accuracy                           0.98       400
   macro avg       0.77      0.79      0.78       400
weighted avg       0.98      0.98      0.98       400


--- DISCRIMINATION ---
              precision    recall  f1-score   support

           0       0.95      0.95      0.95       375
           1       0.23      0.24      0.24        25

    accuracy                           0.90       400
   macro avg       0.59      0.59      0.59       400
weighted avg       0.90      0.90      0.90       400

--- TOXIC ---
        pred_0  pred_1
true_0      78      32
true_1      54     236

--- INSULT ---
        pred_0  pred_1
true_0     195      42
true_1      52     111

--- THREAT ---
        pred_0  pred_1
true_0     385       5
true_1       4       6

--- DISCRIMINATION ---
        pred_0  pred_1
true_0     355      20
true_1      19       6

TO IMPROVE:
- get more data for threat and discrimination
    - oversample these classes?
- Try a different linear model --> LinearSVC does slightly better on text

'''