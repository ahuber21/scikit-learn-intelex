import json
import warnings
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import shap

import xgboost as xgb
from sklearn.base import check_array
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from daal4py import gbt_regression_prediction, get_gbt_model_from_xgboost

X, y = fetch_california_housing(return_X_y=True)
X = X[:, 7:]  # delete some features


def run(random_state):
    print(f"{random_state=}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    xgb_model = xgb.XGBRegressor(max_depth=4, n_estimators=1, random_state=random_state)
    xgb_model.fit(X_train, y_train)
    booster = xgb_model.get_booster()

    d4p_model = get_gbt_model_from_xgboost(booster)

    def predict(X):
        X = check_array(X, dtype=[np.single, np.double])
        predict_algo = gbt_regression_prediction(fptype="float")
        # predict_result = predict_algo.compute(X, d4p_model, setup=False)
        predict_result = predict_algo.compute(X, d4p_model, False, False, setup=False)
        return predict_result.prediction.ravel()

    sample_size = 1
    head = X_test[:sample_size, :].reshape(sample_size, -1)

    explainer = shap.TreeExplainer(xgb_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_from_explainer = explainer.shap_values(head).reshape(1, -1)
    shap_from_daal4py = predict(head).reshape(1, -1)[:, :-1]

    allclose = np.allclose(shap_from_explainer, shap_from_daal4py)
    print(f"{allclose=}")
    diff = shap_from_explainer - shap_from_daal4py
    if not allclose:
        print(f"{shap_from_explainer=}")
        print(f"{shap_from_daal4py=}")
        print(f"{diff=}")

    return allclose


for seed in range(5_000):
    if not run(seed):
        break
