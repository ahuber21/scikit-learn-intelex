import os
import pickle
import time
from tempfile import NamedTemporaryFile

import lightgbm as lgb
import numpy as np
import psutil
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import daal4py

daal4py.daalinit(224)
print(daal4py.num_threads())


# Step 1: Generate synthetic data
# X, y = make_regression(n_samples=100000, n_features=90, noise=0.1, random_state=42)

# Step 2: Split the data into training and testing sets (70:30 ratio)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# # Step 3: Train the LightGBM regression model
# params = {
#     "num_leaves": 200,
#     "objective": "regression",
#     "n_estimators": 200,
#     "verbosity": -1,
#     "max_depth": 6,
#     "max_leaves": 256,
#     "max_bin": 256,
#     "seed": 42,
# }
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# model = lgb.train(
#     params,
#     lgb_train,
#     valid_sets=lgb_eval,
# )


# Step 1: Generate synthetic data with 10 classes
X, y = make_classification(
    n_samples=1000000,
    n_features=28,
    n_informative=28,
    n_redundant=0,
    n_classes=2,
    random_state=42,
)

# Step 2: Split the data into training and testing sets (70:30 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Step 3: Train the LightGBM classification model
params = {
    "verbosity": -1,
    "max_depth": 8,
    "reg-alpha": 0.9,
    "scale-pos-weight": 2,
    "learning-rate": 0.1,
    "subsample": 1,
    "req-lambda": 1,
    "min-child-weight": 0,
    "max_leaves": 256,
    "max_bin": 256,
    "objective": "binary",
    "seed": 42,
    "n_estimators": 3000,
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


def run(depth):
    params["max_depth"] = depth
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
    )

    df = model.trees_to_dataframe()
    depth_stats = df.groupby("tree_index", as_index=False)["node_depth"].max()

    X_timing = X
    while X_timing.shape[0] < 1e6:
        X_timing = np.concatenate([X_timing, X_timing])

    start_time = time.time()
    model.predict(X_timing)
    pred_time = time.time() - start_time

    dmodel = daal4py.mb.convert_model(model)
    with NamedTemporaryFile(mode="wb") as fp:
        pickle.dump(dmodel, fp)
        fp.flush()
        dump_size = os.path.getsize(fp.name)

    start_time = time.time()
    dmodel.predict(X_timing)
    pred_time_d4p = time.time() - start_time

    print(f"\n--------------- depth = {depth} --------------")
    print(f"{X_timing.shape=}")
    print(f"average tree depth:        {np.mean(depth_stats['node_depth'])}")
    print(f"Model dump size:           {dump_size} bytes")
    print(f"Prediction time:           {pred_time} seconds")
    print(f"Prediction time (daal4py): {pred_time_d4p} seconds")
    print(f"Speedup:                   {pred_time/pred_time_d4p:.3}")
    print("----------------------------------------------")


for depth in [8, 10, 12, 15, 18, 22, 25]:
    run(depth)
