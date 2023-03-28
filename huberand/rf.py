ROWS = 500_000
COLS = 10

from time import perf_counter


def fit(clf, X, y):
    start = perf_counter()
    clf.fit(X, y)
    print(f"Took {perf_counter() - start:.2f} seconds")


from sklearn.datasets import make_classification

print("Generate data")
X, y = make_classification(n_samples=ROWS, n_features=COLS,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
X_train, y_train = X[:ROWS], y[:ROWS]
# X_test, y_test = X[100_000:], y[100_000:]

params = {
    "max_depth": 3,
    "n_estimators": 100,
    "random_state": 0,
    "max_features": 1.0,
    "n_jobs": -1
}

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.ensemble import RandomForestClassifier as RFStock

from daal4py.sklearn.ensemble import RandomForestClassifier as RFOptimized

print("Train optimized")
clf_opt = RFOptimized(**params)
fit(clf_opt, X_train, y_train)

# print("Train stock")
# clf = RFStock(**params)
# fit(clf, X_train, y_train)

# print([e.tree_.max_depth for e in clf.estimators_])

# print("Training done.")
