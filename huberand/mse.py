from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFStock
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from daal4py.sklearn.ensemble import RandomForestClassifier as RFOptimized


def fit(clf, X, y):
    start = perf_counter()
    clf.fit(X, y)
    print(f"Took {perf_counter() - start:.2f} seconds")


data, target = pd.read_pickle("data.pkl"), pd.read_pickle("target.pkl")
data = data.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(
    data, target, test_size=0.15, random_state=1
)

params = {
    "n_estimators": 1,
    "n_jobs": -1,
}

print("Train optimized")
clf_opt = RFOptimized(**params)
fit(clf_opt, X_train, y_train)

prediction = clf_opt.predict(X_val)
print("Opt: Confusion matrix")
print(confusion_matrix(y_val, prediction))
print("Opt: Accuracy score")
print(accuracy_score(y_val, prediction))

print("Train stock")
clf = RFStock(**params)
fit(clf, X_train, y_train)

prediction = clf.predict(X_val)
print("Stock: Confusion matrix")
print(confusion_matrix(y_val, prediction))
print("Stock: Accuracy score")
print(accuracy_score(y_val, prediction))

print("Training done.")
