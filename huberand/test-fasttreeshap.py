import warnings
from time import time

import fasttreeshap
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import daal4py

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import shap

X, y = fetch_california_housing(return_X_y=True)
# X = X[:, 6:]  # drop some features for debugging
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
xgb_model = xgb.XGBRegressor(max_depth=7, n_estimators=10, random_state=3)
print(f"{X_train.shape=}")
print("Running fit...")
xgb_model.fit(X_train, y_train)
booster = xgb_model.get_booster()

d4p_model = daal4py.mb.convert_model(booster)

sample_size = 1
head = X_test[:sample_size, :].reshape(sample_size, -1)

print(f"{head.shape=}")

shap_explainer = shap.TreeExplainer(xgb_model, algorithm="auto")
fasttreeshap_explainer = fasttreeshap.TreeExplainer(xgb_model, algorithm="v2_2")

# time it - interactions
if False:
    duplications = 7

    X_timing = X_test
    for _ in range(duplications - 1):
        X_timing = np.concatenate([X_timing, X_timing])

    print(f"{X_test.shape=}")
    print(f"{X_timing.shape=}")

    # import os
    # input(f"ENTER for action - PID {os.getpid()}")
    # print("Starting measurement")

    start_time = time()
    d4p_values = d4p_model.predict(X_timing, pred_interactions=True)
    # predict_algo.compute(X_timing, d4p_model)
    daal4py_time = time() - start_time
    print(f"{d4p_values.shape=}")
    print(f"{daal4py_time=:.2f} s")

    start_time = time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = shap_explainer.shap_interaction_values(X_timing)
    shap_time = time() - start_time
    print(f"{shap_values.shape=}")
    print(f"{shap_time=:.2f} s")

    speedup = shap_time / daal4py_time

    start_time = time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = fasttreeshap_explainer.shap_interaction_values(X_timing)
    fasttreeshap_time = time() - start_time
    print(f"{shap_values.shape=}")
    print(f"{fasttreeshap_time=:.2f} s")

    print(f"{shap_time / daal4py_time=}")
    print(f"{fasttreeshap_time / daal4py_time=}")


# check correctness - contributions
if True:
    # predict from booster
    shap_from_daal4py = d4p_model.predict(
        head, pred_contribs=True, pred_interactions=False
    )[:, :-1]
    print(f"{shap_from_daal4py=}")

    shap_from_fasttreeshap = fasttreeshap_explainer(head, check_additivity=False).values
    print(f"{shap_from_fasttreeshap=}")

    print(f"\n\n{np.allclose(shap_from_fasttreeshap, shap_from_daal4py, atol=1e-5)=}")
    print(f"{np.absolute(shap_from_fasttreeshap - shap_from_daal4py)=}")

    print(
        f"Largest deviation: {np.absolute(shap_from_fasttreeshap - shap_from_daal4py).reshape(1, -1).max():1.3e}"
    )


"""
b tree_shap.h:1952
r test-fasttreeshap.py
p max_leaves * max_combinations * trees.tree_limit
"""
