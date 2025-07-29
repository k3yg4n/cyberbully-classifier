import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import make_scorer, f1_score, accuracy_score

df = pd.read_csv("cyberbully-classifier/data/output/tfidf_dataset.csv")

# split features and labels
X = df.drop(columns=["cyberbullying"])
y = df["cyberbullying"]

# apply Truncated SVD (LSA)
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

# set up 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')
acc_scorer = make_scorer(accuracy_score)

# evaluate weighted KNN with k from 1 to 24
f1_scores_all = []
acc_scores_all = []

for k in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    f1_scores = cross_val_score(knn, X_reduced, y, cv=kf, scoring=f1_scorer)
    acc_scores = cross_val_score(knn, X_reduced, y, cv=kf, scoring=acc_scorer)

    f1_scores_all.append(f1_scores.mean())
    acc_scores_all.append(acc_scores.mean())

    print(f"\nk = {k}")
    print(f"Mean F1-Weighted Score: {f1_scores.mean():.4f}")
    print(f"Mean Accuracy         : {acc_scores.mean():.4f}")

# k = 1
# Mean F1-Weighted Score: 0.7526
# Mean Accuracy         : 0.7532

# k = 2
# Mean F1-Weighted Score: 0.7521
# Mean Accuracy         : 0.7527

# k = 3
# Mean F1-Weighted Score: 0.7697
# Mean Accuracy         : 0.7708

# k = 4
# Mean F1-Weighted Score: 0.7710
# Mean Accuracy         : 0.7720

# k = 5
# Mean F1-Weighted Score: 0.7768
# Mean Accuracy         : 0.7781

# k = 6
# Mean F1-Weighted Score: 0.7783
# Mean Accuracy         : 0.7799

# k = 7
# Mean F1-Weighted Score: 0.7800
# Mean Accuracy         : 0.7816

# k = 8
# Mean F1-Weighted Score: 0.7810
# Mean Accuracy         : 0.7827

# k = 9
# Mean F1-Weighted Score: 0.7810
# Mean Accuracy         : 0.7828

# k = 10
# Mean F1-Weighted Score: 0.7808
# Mean Accuracy         : 0.7827

# k = 11
# Mean F1-Weighted Score: 0.7817
# Mean Accuracy         : 0.7837

# k = 12
# Mean F1-Weighted Score: 0.7818
# Mean Accuracy         : 0.7838

# k = 13
# Mean F1-Weighted Score: 0.7809
# Mean Accuracy         : 0.7829

# k = 14
# Mean F1-Weighted Score: 0.7824
# Mean Accuracy         : 0.7845

# k = 15
# Mean F1-Weighted Score: 0.7801
# Mean Accuracy         : 0.7822

# k = 16
# Mean F1-Weighted Score: 0.7812
# Mean Accuracy         : 0.7835

# k = 17
# Mean F1-Weighted Score: 0.7799
# Mean Accuracy         : 0.7822

# k = 18
# Mean F1-Weighted Score: 0.7806
# Mean Accuracy         : 0.7830

# k = 19
# Mean F1-Weighted Score: 0.7801
# Mean Accuracy         : 0.7825

# k = 20
# Mean F1-Weighted Score: 0.7813
# Mean Accuracy         : 0.7839

# k = 21
# Mean F1-Weighted Score: 0.7801
# Mean Accuracy         : 0.7827

# k = 22
# Mean F1-Weighted Score: 0.7820
# Mean Accuracy         : 0.7847

# k = 23
# Mean F1-Weighted Score: 0.7813
# Mean Accuracy         : 0.7841

# k = 24
# Mean F1-Weighted Score: 0.7808
# Mean Accuracy         : 0.7836