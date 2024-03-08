import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

import daal4py as d4p


def test_xgb_classification():
    X, y = make_classification(
        n_samples=500, n_classes=2, n_features=15, n_informative=10, random_state=42
    )
    X_test = X[:2, :]
    xgb_model = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=50,
        random_state=42,
        base_score=0.7,
        objective="binary:logitraw",
    )
    xgb_model.fit(X, y)

    print("Start model building")
    start = time.time()

    d4p_model = d4p.mb.convert_model(xgb_model.get_booster())

    print(f"Model created in {time.time() - start:.2f} s")
    # np.testing.assert_allclose(xgb_model.predict_proba(X_test), d4p_model.predict_proba(X_test), rtol=1e-5)


if __name__ == "__main__":
    test_xgb_classification()
