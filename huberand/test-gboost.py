import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

import daal4py as d4p

# Datasets creation
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)

# training parameters setting
params = {
    "max_bin": 256,
    "scale_pos_weight": 2,
    "lambda_l2": 1,
    "alpha": 0.9,
    "max_depth": 6,
    "num_leaves": 2**6,
    "verbosity": 0,
    "objective": "multi:softmax",
    "learning_rate": 0.3,
    "num_class": 5,
    "n_estimators": 25,
}

# Training
xgb_model = xgb.train(params, xgb_train, num_boost_round=100)

# print(hasattr(xgb_model, "best_iteration"))
# print(xgb_model.best_iteration)

# Conversion to daal4py
daal_model = d4p.mb.convert_model(xgb_model)
