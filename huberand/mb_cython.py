from multiprocessing import Manager

import XGBoost as xgb
from mb_cython_module import process_trees_parallel
from sklearn.datasets import make_regression

from daal4py import gbt_clf_model_builder, gbt_reg_model_builder


def test_booster(booster):
    # Assuming you have a list of raw trees in 'dump'
    dump = booster.get_dump(dump_format="json", with_stats=True)

    # Set the number of trees to process
    max_trees = len(dump)  # Process all trees

    # Create a shared list to store the processed root nodes
    with Manager() as manager:
        shared_root_nodes = manager.list([None] * max_trees)

        # Call the Cython function for parallel processing
        process_trees_parallel(max_trees, dump, shared_root_nodes)

        # Convert the shared list to a regular Python list
        root_nodes_list = list(shared_root_nodes)

    # Now you have a list of root nodes to work with
    for root_node in root_nodes_list:
        # Process each root node as needed
        pass


def test_xgb_regressor():
    print("Gen data...")
    X, y = make_regression(random_state=3)

    xgb_model = xgb.XGBRegressor(max_depth=5, n_estimators=50, random_state=3)
    print(f"{X.shape=}")
    print("Running fit...")
    xgb_model.fit(X, y)

    booster = xgb_model.get_booster()

    print("Start model building")
    start = time.time()
    d4p_model = get_gbt_model_from_xgboost(booster)
    print(f"Model created in {time.time() - start:.2f} s")


if __name__ == "__main__":
    test_xgb_regressor()
