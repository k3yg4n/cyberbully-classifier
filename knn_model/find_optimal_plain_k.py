import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score

# load preprocessed TF-IDF dataset
df = pd.read_csv("cyberbully-classifier/data/output/tfidf_dataset.csv")

# split features and label
X = df.drop(columns=['cyberbullying'])  # TF-IDF features
y = df['cyberbullying']                 # 0 = Not cyberbullying, 1 = Cyberbullying

# set up 5-fold cross validation
# using 5-folds across all the models to standardize evaluation 
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# evaluate plain KNN with k from 1 to 12
f1_scorer = make_scorer(f1_score, average='weighted')
acc_scorer = make_scorer(accuracy_score)

for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    f1_scores = cross_val_score(knn, X, y, cv=kf, scoring=f1_scorer)
    acc_scores = cross_val_score(knn, X, y, cv=kf, scoring=acc_scorer)
    print(f"\nk = {k}")
    print(f"F1-Weighted Scores: {f1_scores}")
    print(f"Accuracy Scores   : {acc_scores}")
    print(f"Mean F1-Weighted Score: {f1_scores.mean():.4f}")
    print(f"Mean Accuracy         : {acc_scores.mean():.4f}")

# k = 1
# F1-Weighted Scores: [0.62697683 0.62782604 0.51355572 0.6434419  0.62968368]
# Accuracy Scores   : [0.65654854 0.6587057  0.5761171  0.67118644 0.65978428]
# Mean F1-Weighted Score: 0.6083
# Mean Accuracy         : 0.6445

# k = 2

# k = 3
# F1-Weighted Scores: [0.48872426 0.48614654 0.46444081 0.49087749 0.49632023]
# Accuracy Scores   : [0.56409861 0.56070878 0.55855162 0.56533128 0.56748844]
# Mean F1-Weighted Score: 0.4853
# Mean Accuracy         : 0.5632

# k = 4
# F1-Weighted Scores: [0.48123194 0.48241096 0.46621005 0.48478827 0.49475842]
# Accuracy Scores   : [0.5596302  0.55824345 0.55793529 0.55916795 0.56702619]
# Mean F1-Weighted Score: 0.4819
# Mean Accuracy         : 0.5604

# k = 5
# F1-Weighted Scores: [0.48725695 0.48527837 0.459556   0.48705977 0.49530558]
# Accuracy Scores   : [0.56548536 0.5614792  0.55793529 0.56456086 0.56702619]
# Mean F1-Weighted Score: 0.4829
# Mean Accuracy         : 0.5633

# k = 6
# F1-Weighted Scores: [0.59311961 0.58388318 0.51055145 0.59941504 0.58976341]
# Accuracy Scores   : [0.63636364 0.63189522 0.57580894 0.64298921 0.6357473 ]
# Mean F1-Weighted Score: 0.5753
# Mean Accuracy         : 0.6246

# k = 7
# F1-Weighted Scores: [0.49164157 0.49520389 0.44565634 0.60456901 0.5973749 ]
# Accuracy Scores   : [0.56779661 0.56733436 0.55208012 0.6440678  0.63959938]
# Mean F1-Weighted Score: 0.5269
# Mean Accuracy         : 0.5942

# k = 8
# F1-Weighted Scores: [0.48886642 0.58658437 0.45265382 0.59477309 0.5884709 ]
# Accuracy Scores   : [0.56409861 0.63281972 0.55469954 0.63959938 0.63451464]
# Mean F1-Weighted Score: 0.5423
# Mean Accuracy         : 0.6051

# k = 9
# F1-Weighted Scores: [0.48511111 0.49085565 0.44888293 0.4876254  0.5917184 ]
# Accuracy Scores   : [0.56579353 0.56533128 0.55423729 0.56394453 0.63651772]
# Mean F1-Weighted Score: 0.5008
# Mean Accuracy         : 0.5772

# k = 10
# F1-Weighted Scores: [0.59297642 0.49215775 0.50689576 0.60706737 0.526802  ]
# Accuracy Scores   : [0.6357473  0.5642527  0.57580894 0.64853621 0.59845917]
# Mean F1-Weighted Score: 0.5452
# Mean Accuracy         : 0.6046

# k = 11
# F1-Weighted Scores: [0.48667189 0.44699492 0.44906804 0.48408308 0.5870451 ]
# Accuracy Scores   : [0.56302003 0.54730354 0.55423729 0.5624037  0.63359014]
# Mean F1-Weighted Score: 0.4908
# Mean Accuracy         : 0.5721