from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFStock
from sklearn.model_selection import train_test_split

from daal4py.sklearn.ensemble import RandomForestClassifier as RFOptimized

CSV_NAME = "max_features_times.csv"

# load the data prepared in prepare_data.py
data, target = pd.read_pickle("data.pkl"), pd.read_pickle("target.pkl")
data = data.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.15, random_state=1)

index_names = []
times_stock = []
times_opt = []

def fit(X, y, max_features):
    params = {
        "max_features": max_features,
        "max_depth": 10,
        "n_estimators": 300,
        "random_state": 0,
        "n_jobs": -1
    }

    clf = RFStock(**params)
    clf_opt = RFOptimized(**params)

    print(f"Training stock with {max_features=}")
    start = perf_counter()
    clf.fit(X, y)
    t_stock = perf_counter() - start
    print(f"Took {t_stock=}")

    print(f"Training opt with {max_features=}")
    start = perf_counter()
    clf_opt.fit(X, y)
    t_opt = perf_counter() - start
    print(f"Took {t_opt=}")

    index_names.append(f"{max_features=}")
    times_stock.append(t_stock)
    times_opt.append(t_opt)

    df = pd.DataFrame({"stock": times_stock, "optimized": times_opt}, index=index_names)
    df.to_csv(CSV_NAME)
    print(f"Updated {CSV_NAME}")

for max_features in np.linspace(0.1, 1.0, 19):
    fit(X_train, y_train, max_features)
