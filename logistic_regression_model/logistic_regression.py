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
STRATIFY_LABEL = 'threat' # Stratify to ensure the train/test splits have the same relative frequency of toxic vs. non‑toxic

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

Evaluating on test set: 

--- TOXIC ---
              precision    recall  f1-score   support

           0       0.94      0.89      0.91       106
           1       0.85      0.92      0.88        73

    accuracy                           0.90       179
   macro avg       0.89      0.90      0.90       179
weighted avg       0.90      0.90      0.90       179


--- INSULT ---
              precision    recall  f1-score   support

           0       0.87      0.94      0.90       118
           1       0.86      0.72      0.79        61

    accuracy                           0.87       179
   macro avg       0.86      0.83      0.84       179
weighted avg       0.87      0.87      0.86       179


--- THREAT ---
              precision    recall  f1-score   support

           0       0.94      0.97      0.96       156
           1       0.78      0.61      0.68        23

    accuracy                           0.93       179
   macro avg       0.86      0.79      0.82       179
weighted avg       0.92      0.93      0.92       179


--- DISCRIMINATION ---
              precision    recall  f1-score   support

           0       0.97      0.94      0.96       159
           1       0.62      0.75      0.68        20

    accuracy                           0.92       179
   macro avg       0.80      0.85      0.82       179
weighted avg       0.93      0.92      0.92       179


--- TOXIC ---
        pred_0  pred_1
true_0      94      12
true_1       6      67

--- INSULT ---
        pred_0  pred_1
true_0     111       7
true_1      17      44

--- THREAT ---
        pred_0  pred_1
true_0     152       4
true_1       9      14

--- DISCRIMINATION ---
        pred_0  pred_1
true_0     150       9
true_1       5      15

TO IMPROVE:
- get more data for threat and discrimination
    - oversample these classes?
- Try a different linear model --> LinearSVC does slightly better on text

'''