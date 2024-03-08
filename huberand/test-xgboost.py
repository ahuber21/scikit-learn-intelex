import json
import warnings
from pathlib import Path
from time import time

import fasttreeshap
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import daal4py

X, y = fetch_california_housing(return_X_y=True)
# X = X[:, :2]  # drop some features for debugging
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
xgb_model = xgb.XGBRegressor(max_depth=12, n_estimators=50, random_state=3)
print(f"{X_train.shape=}")
print("Running fit...")
xgb_model.fit(X_train, y_train)
booster = xgb_model.get_booster()

# measure model conversion time
if False:
    import os

    input(f"ready? {os.getpid()}")
    for _ in range(15):
        d4p_model = daal4py.mb.convert_model(booster)
    exit()

d4p_model = daal4py.mb.convert_model(booster)

# draw tree
if False:
    predict_algo = gbt_regression_prediction(fptype="float")
    predict_result = predict_algo.compute(X, d4p_model)

    tree = eval(xgb_model.get_booster().get_dump(dump_format="json", with_stats=True)[0])
    wd = Path(__file__).parent
    import os

    import matplotlib.pyplot as plt

    os.environ["PATH"] = (
        "/export/users/huberand/miniconda3/envs/build/bin:" + os.environ["PATH"]
    )
    fig, ax = plt.subplots(figsize=(120, 50))
    ax = xgb.plot_tree(booster, num_trees=0, ax=ax)
    plt.savefig(wd / "tree.png")
    with open(wd / "tree.json", "w") as fp:
        json.dump(tree, fp, indent=2, sort_keys=True)


sample_size = 1
head = X_test[:sample_size, :].reshape(sample_size, -1)

print(f"{head.shape=}")
# print(f"{head=}")

if False:
    print("Running predict...")
    # pred =d4p_model.predict(head).reshape(sample_size, -1)
    # print(f"{pred.shape=}")
    # print(f"{pred=}")
    print(f"{xgb_model.predict(head)=}")


# explainer = shap.KernelExplainer(predict, shap.sample(X_train))
# print(explainer.shap_values(X_test))


# model = {
#     trees = d4p_model.
# }
explainer = fasttreeshap.TreeExplainer(xgb_model)

# check correctness - contributions
if True:
    # predict from booster
    shap_from_daal4py = d4p_model.predict(
        head, pred_contribs=True, pred_interactions=False
    )
    print(f"{shap_from_daal4py=}")

    shap_from_booster = booster.predict(
        xgb.DMatrix(head),
        pred_contribs=True,
        pred_interactions=False,
        approx_contribs=False,
        validate_features=False,
    )
    print(f"{shap_from_booster=}")

    print(f"\n\n{np.allclose(shap_from_booster, shap_from_daal4py, atol=1e-5)=}")
    print(f"{np.absolute(shap_from_booster - shap_from_daal4py)=}")

    print(
        f"Largest deviation: {np.absolute(shap_from_booster - shap_from_daal4py).reshape(1, -1).max():1.3e}"
    )


# check correctness - interactions
if False:
    # predict from booster
    shap_from_daal4py = d4p_model.predict(
        head, pred_contribs=False, pred_interactions=True
    )
    print(f"{shap_from_daal4py=}")

    shap_from_booster = booster.predict(
        xgb.DMatrix(head),
        pred_contribs=False,
        pred_interactions=True,
        approx_contribs=False,
        validate_features=False,
    )
    print(f"{shap_from_booster=}")

    print(f"\n\n{np.allclose(shap_from_booster, shap_from_daal4py, atol=1e-5)=}")
    # print(f"{np.absolute(shap_from_booster - shap_from_daal4py)=}")

    print(
        f"Largest deviation: {np.absolute(shap_from_booster - shap_from_daal4py).reshape(1, -1).max():1.3e}"
    )


# sudo sysctl -w kernel.yama.ptrace_scope=0

# time it - contributions
if True:
    duplications = 8

    X_timing = X_test
    for _ in range(duplications - 1):
        X_timing = np.concatenate([X_timing, X_timing])

    print(f"{X_test.shape=}")
    print(f"{X_timing.shape=}")

    import os

    input(f"ENTER for action - PID {os.getpid()}")
    print("Starting measurement")

    start_time = time()
    d4p_model.predict(X_timing, pred_contribs=True)
    # predict_algo.compute(X_timing, d4p_model)
    daal4py_time = time() - start_time
    print(f"{daal4py_time=:.2f} s")

    # start_time = time()
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     explainer.shap_values(X_timing, check_additivity=False)
    # shap_time = time() - start_time
    # print(f"{shap_time=:.2f} s")

    # speedup = shap_time / daal4py_time
    # print(f"{speedup=}")

# time it - interactions
if False:
    duplications = 3

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
        shap_values = explainer.shap_interaction_values(X_timing)
    shap_time = time() - start_time
    print(f"{shap_values.shape=}")
    print(f"{shap_time=:.2f} s")

    speedup = shap_time / daal4py_time
    print(f"{speedup=}")


"""

b gbt_regression_predict_dense_default_batch_impl.i:427
b gbt_regression_predict_dense_default_batch_impl.i:354
b gbt_regression_predict_dense_default_batch_impl.i:489

x/60xw 0x5555592acad0

(gdb) p resBD.get()
$1 = (float *) 0x5555596fc180
x/18xw 0x5555596fc180



"""
