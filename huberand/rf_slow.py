## OPTION 1: Load from OpenML (unpacking is very slow)
# from sklearn.datasets import fetch_openml

# data, target = fetch_openml(data_id=42759, return_X_y=True, data_home="../scikit_learn_data")
# categorical_features = list(set(data.columns) - set(data._get_numeric_data().columns))
# data = data.drop(columns=categorical_features)
# data.to_pickle("data.pkl")
# target.to_pickle("target.pkl")

## OPTION 2: Load pickle saved when using OPTION 1
import pandas as pd

data, target = pd.read_pickle("data.pkl"), pd.read_pickle("target.pkl")
data = data.fillna(0)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    data, target, test_size=0.15, random_state=1
)

from time import perf_counter


def fit(clf, X, y):
    start = perf_counter()
    clf.fit(X, y)
    print(f"Took {perf_counter() - start:.2f} seconds")


# params = {
#     "max_depth": 3,
#     "n_estimators": 300,
#     "random_state": 0,
#     "max_features": 1.0,
#     "n_jobs": -1
# }

# params = {
#     "n_estimators": 50,
#     "n_jobs": -1
# }

params = {
    "n_estimators": 100,
    "n_jobs": -1,
    # "max_features": 0.4,
    # "useConstFeatures": False,
}

from sklearn.ensemble import RandomForestClassifier as RFStock

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.metrics import accuracy_score, confusion_matrix

from daal4py.sklearn.ensemble import RandomForestClassifier as RFOptimized

# input("Press any button to continue")

print("Train optimized")
clf_opt = RFOptimized(**params)
fit(clf_opt, X_train, y_train)

prediction = clf_opt.predict(X_val)
print("Opt: Confusion matrix")
print(confusion_matrix(y_val, prediction))
print("Opt: Accuracy score")
print(accuracy_score(y_val, prediction))

print([e.tree_.max_depth for e in clf_opt.estimators_])


print("Train stock")
del params["useConstFeatures"]
clf = RFStock(**params)
fit(clf, X_train, y_train)

prediction = clf.predict(X_val)
print("Stock: Confusion matrix")
print(confusion_matrix(y_val, prediction))
print("Stock: Accuracy score")
print(accuracy_score(y_val, prediction))

print([e.tree_.max_depth for e in clf.estimators_])

print("Training done.")
