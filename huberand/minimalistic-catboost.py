import catboost as cb
import numpy as np
from mytest import get_gbt_model_from_catboost
from sklearn.datasets import make_regression

import daal4py as d4p

d4p.get_gbt_model_from_catboost = get_gbt_model_from_catboost

X, y = make_regression(n_samples=100, n_features=2, random_state=3)

print(f"{X.shape=}")
print("Running fit...")

model = cb.CatBoostRegressor(max_depth=3, n_estimators=1)  # GOOD
# model = cb.CatBoostRegressor(max_depth=5, n_estimators=1)  # BAD
duplications = 1
X = np.concatenate([X] * duplications)
y = np.concatenate([y] * duplications)
model.fit(X, y)

d4p_model = d4p.mb.convert_model(model)

d4p_predict = d4p_model.predict(X[:1, :])
predict = model.predict(X[:1, :])
print(f"d4p pred:   {d4p_predict}")
print(f"cb pred:    {predict}")

d4p_contribs = d4p_model.predict(X[:1, :], pred_contribs=True)
cb_contribs = model.get_feature_importance(
    cb.Pool(X[:1, :], y[:1]), type="ShapValues", shap_calc_type="Exact"
)

print(f"d4p:   {d4p_contribs}")
print(f"cb:    {cb_contribs}")

print(f"delta: {d4p_contribs[0][0] - cb_contribs[0][0]}")
