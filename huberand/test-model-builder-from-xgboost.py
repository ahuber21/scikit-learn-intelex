import json
import pickle
import time
from collections import deque
from tempfile import NamedTemporaryFile
from typing import Any, Deque, Dict, List, Optional, Tuple
from warnings import warn

import catboost as cb
import lightgbm as lgbm
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

import daal4py
from daal4py import gbt_clf_model_builder, gbt_reg_model_builder


class Node:
    """Helper class holding Tree Node information"""

    def __init__(
        self,
        cover: float,
        is_leaf: bool,
        default_left: bool,
        feature: int,
        value: float,
        n_children: int = 0,
        left_child: "Optional[Node]" = None,
        right_child: "Optional[Node]" = None,
        parent_id: Optional[int] = -1,
        position: Optional[int] = -1,
    ) -> None:
        self.cover = cover
        self.is_leaf = is_leaf
        self.default_left = default_left
        self.__feature = feature
        self.value = value
        self.n_children = n_children
        self.left_child = left_child
        self.right_child = right_child
        self.parent_id = parent_id
        self.position = position

    @staticmethod
    def from_xgb_dict(input_dict: Dict[str, Any]) -> "Node":
        if "children" in input_dict:
            left_child = Node.from_xgb_dict(input_dict["children"][0])
            right_child = Node.from_xgb_dict(input_dict["children"][1])
            n_children = 2 + left_child.n_children + right_child.n_children
        else:
            left_child = None
            right_child = None
            n_children = 0
        is_leaf = "leaf" in input_dict
        default_left = "yes" in input_dict and input_dict["yes"] == input_dict["missing"]
        return Node(
            cover=input_dict["cover"],
            is_leaf=is_leaf,
            default_left=default_left,
            feature=input_dict.get("split"),
            value=input_dict["leaf"] if is_leaf else input_dict["split_condition"],
            n_children=n_children,
            left_child=left_child,
            right_child=right_child,
        )

    @staticmethod
    def from_lightgbm_dict(input_dict: Dict[str, Any]) -> "Node":
        if "tree_structure" in input_dict:
            tree = input_dict["tree_structure"]
        else:
            tree = input_dict

        n_children = 0
        if "left_child" in tree:
            left_child = Node.from_lightgbm_dict(tree["left_child"])
            n_children += 1 + left_child.n_children
        else:
            left_child = None
        if "right_child" in tree:
            right_child = Node.from_lightgbm_dict(tree["right_child"])
            n_children += 1 + right_child.n_children
        else:
            right_child = None

        is_leaf = "leaf_value" in tree
        empty_leaf = is_leaf and "leaf_count" not in tree
        if is_leaf:
            cover = tree["leaf_count"]
        else:
            cover = tree["internal_count"]
        return Node(
            cover=cover,
            is_leaf=is_leaf,
            default_left=is_leaf or tree["default_left"],
            feature=tree.get("split_feature"),
            value=tree["leaf_value"] if is_leaf else tree["threshold"],
            n_children=n_children,
            left_child=left_child,
            right_child=right_child,
        )

    def get_value_closest_float_downward(self) -> np.float64:
        """Get the closest exact fp value smaller than self.value"""
        # return np.nextafter(np.single(self.value), np.single(-np.inf))
        return self.value

    def get_children(self) -> "Optional[Tuple[Node, Node]]":
        if not self.left_child or not self.right_child:
            assert self.is_leaf
        else:
            return (self.left_child, self.right_child)

    @property
    def feature(self) -> int:
        if isinstance(self.__feature, int):
            return self.__feature
        if isinstance(self.__feature, str) and self.__feature.isnumeric():
            return int(self.__feature)
        raise ValueError(
            f"Feature names must be integers (got ({type(self.__feature)}){self.__feature})"
        )


class TreeView:
    """Helper class, treating a list of nodes as one tree"""

    def __init__(self, root_node: Node) -> None:
        self.root_node = root_node
        self.tree_id = -1

    @property
    def is_leaf(self) -> bool:
        return self.root_node.is_leaf

    @property
    def value(self) -> float:
        if not self.is_leaf:
            raise ValueError("Tree is not a leaf-only tree")
        if not self.root_node.value:
            raise ValueError("Tree is leaf-only but leaf node has no value")
        return self.root_node.value

    @property
    def cover(self) -> float:
        if not self.is_leaf:
            raise ValueError("Tree is not a leaf-only tree")
        return self.root_node.cover

    @property
    def n_nodes(self) -> int:
        return self.root_node.n_children + 1


class TreeList(list):
    """Helper class that is able to extract all information required by the
    model builders from various objects"""

    @staticmethod
    def from_xgb_booster(booster, max_trees: int) -> "TreeList":
        """
        Load a TreeList from an xgb.Booster object
        Note: We cannot type-hint the xgb.Booster without loading xgb as dependency in pyx code,
              therefore not type hint is added.
        """
        tl = TreeList()
        dump = booster.get_dump(dump_format="json", with_stats=True)
        for tree_id, raw_tree in enumerate(dump):
            if max_trees > 0 and tree_id == max_trees:
                break
            raw_tree_parsed = json.loads(raw_tree)
            root_node = Node.from_xgb_dict(raw_tree_parsed)
            tl.append(TreeView(root_node=root_node))

        return tl

    @staticmethod
    def from_lightgbm_booster_dump(dump: Dict[str, Any]) -> "TreeList":
        """
        Load a TreeList from a lgbm.Model object
        Note: We cannot type-hint the the Model without loading lightgbm as dependency in pyx code,
              therefore not type hint is added.
        """
        tl = TreeList()
        for tree_dict in dump["tree_info"]:
            root_node = Node.from_lightgbm_dict(tree_dict)
            tl.append(TreeView(root_node=root_node))

        return tl

    def __setitem__(self):
        raise NotImplementedError(
            "Use TreeList.from_*() methods to initialize a TreeList"
        )


def get_gbt_model_from_tree_list(
    tree_list: TreeList,
    n_iterations: int,
    is_regression: bool,
    n_features: int,
    n_classes: int,
    base_score: Optional[float] = None,
):
    """Return a GBT Model from TreeList"""

    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )

    if is_regression:
        for tree in tree_list:
            tree.tree_id = mb.create_tree(tree.n_nodes)
    else:
        class_label = 0
        for counter, tree in enumerate(tree_list, start=1):
            tree.tree_id = mb.create_tree(n_nodes=tree.n_nodes, class_label=class_label)
            if counter % n_iterations == 0:
                class_label += 1

    for tree in tree_list:
        if tree.is_leaf:
            mb.add_leaf(tree_id=tree.tree_id, response=tree.value, cover=tree.cover)
            continue

        root_node = tree.root_node
        parent_id = mb.add_split(
            tree_id=tree.tree_id,
            feature_index=root_node.feature,
            feature_value=root_node.get_value_closest_float_downward(),
            cover=root_node.cover,
            default_left=root_node.default_left,
        )

        # create queue
        node_queue: Deque[Node] = deque()
        children = root_node.get_children()
        assert children is not None
        for position, child in enumerate(children):
            child.parent_id = parent_id
            child.position = position
            node_queue.append(child)

        while node_queue:
            node = node_queue.popleft()
            assert node.parent_id != -1, "node.parent_id must not be -1"
            assert node.position != -1, "node.position must not be -1"

            if node.is_leaf:
                mb.add_leaf(
                    tree_id=tree.tree_id,
                    response=node.get_value_closest_float_downward(),
                    cover=node.cover,
                    parent_id=node.parent_id,
                    position=node.position,
                )
            else:
                parent_id = mb.add_split(
                    tree_id=tree.tree_id,
                    feature_index=node.feature,
                    feature_value=node.get_value_closest_float_downward(),
                    cover=node.cover,
                    default_left=node.default_left,
                    parent_id=node.parent_id,
                    position=node.position,
                )

                children = node.get_children()
                assert children is not None
                for position, child in enumerate(children):
                    child.parent_id = parent_id
                    child.position = position
                    node_queue.append(child)

    import os

    import psutil

    def memory():
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2

    print(f"before mb.model, memory={memory()}")

    model = mb.model(base_score=base_score)

    print(f"after mb.model, memory={memory()}")

    return model


def get_gbt_model_from_lightgbm(model: Any, booster=None) -> Any:
    if booster is None:
        booster = model.dump_model()

    n_features = booster["max_feature_idx"] + 1
    n_iterations = len(booster["tree_info"]) / booster["num_tree_per_iteration"]
    n_classes = booster["num_tree_per_iteration"]

    is_regression = False
    objective_fun = booster["objective"]
    if n_classes > 2:
        if "multiclass" not in objective_fun:
            raise TypeError(
                "multiclass (softmax) objective is only supported for multiclass classification"
            )
    elif "binary" in objective_fun:  # nClasses == 1
        n_classes = 2
    else:
        is_regression = True

    tree_list = TreeList.from_lightgbm_booster_dump(booster)

    return get_gbt_model_from_tree_list(
        tree_list,
        n_iterations=n_iterations,
        is_regression=is_regression,
        n_features=n_features,
        n_classes=n_classes,
    )


def get_xgboost_params(booster):
    return json.loads(booster.save_config())


def get_gbt_model_from_xgboost(booster: Any, xgb_config=None) -> Any:
    # Release Note for XGBoost 1.5.0: Python interface now supports configuring
    # constraints using feature names instead of feature indices. This also
    # helps with pandas input with set feature names.
    booster.feature_names = [str(i) for i in range(booster.num_features())]

    if xgb_config is None:
        xgb_config = get_xgboost_params(booster)

    n_features = int(xgb_config["learner"]["learner_model_param"]["num_feature"])
    n_classes = int(xgb_config["learner"]["learner_model_param"]["num_class"])
    base_score = float(xgb_config["learner"]["learner_model_param"]["base_score"])

    is_regression = False
    objective_fun = xgb_config["learner"]["learner_train_param"]["objective"]
    if n_classes > 2:
        if objective_fun not in ["multi:softprob", "multi:softmax"]:
            raise TypeError(
                "multi:softprob and multi:softmax are only supported for multiclass classification"
            )
    elif objective_fun.startswith("binary:"):
        if objective_fun not in ["binary:logistic", "binary:logitraw"]:
            raise TypeError(
                "only binary:logistic and binary:logitraw are supported for binary classification"
            )
        n_classes = 2
        if objective_fun == "binary:logitraw":
            # daal4py always applies a sigmoid for pred_proba, wheres XGBoost
            # returns raw predictions with logitraw
            warn(
                "objective='binary:logitraw' selected\n"
                "XGBoost returns raw class scores when calling pred_proba()\n"
                "whilst scikit-learn-intelex always uses binary:logistic"
            )
            if base_score != 0.5:
                warn(
                    "objective='binary:logitraw' ignores base_score, fixing base_score to 0.5"
                )
                base_score = 0.5
    else:
        is_regression = True

    # 0 if best_iteration does not exist
    max_trees = getattr(booster, "best_iteration", -1) + 1
    if n_classes > 2:
        max_trees *= n_classes
    tree_list = TreeList.from_xgb_booster(booster, max_trees)

    for num, tree in enumerate(tree_list):
        print(f"tree = {num}, nodes = {tree.root_node.n_children + 1}")

    if hasattr(booster, "best_iteration"):
        n_iterations = booster.best_iteration + 1
    else:
        n_iterations = len(tree_list) // (n_classes if n_classes > 2 else 1)

    return get_gbt_model_from_tree_list(
        tree_list,
        n_iterations=n_iterations,
        is_regression=is_regression,
        n_features=n_features,
        n_classes=n_classes,
        base_score=base_score,
    )


def get_gbt_model_from_xgboost_old(booster: Any, xgb_config=None) -> Any:
    class Node:
        def __init__(self, tree: Dict, parent_id: int, position: int):
            self.tree = tree
            self.parent_id = parent_id
            self.position = position

    # Release Note for XGBoost 1.5.0: Python interface now supports configuring
    # constraints using feature names instead of feature indices. This also
    # helps with pandas input with set feature names.
    lst = [*range(booster.num_features())]
    booster.feature_names = [str(i) for i in lst]

    trees_arr = booster.get_dump(dump_format="json")
    if xgb_config is None:
        xgb_config = get_xgboost_params(booster)

    n_features = int(xgb_config["learner"]["learner_model_param"]["num_feature"])
    n_classes = int(xgb_config["learner"]["learner_model_param"]["num_class"])
    base_score = float(xgb_config["learner"]["learner_model_param"]["base_score"])

    is_regression = False
    objective_fun = xgb_config["learner"]["learner_train_param"]["objective"]
    if n_classes > 2:
        if objective_fun not in ["multi:softprob", "multi:softmax"]:
            raise TypeError(
                "multi:softprob and multi:softmax are only supported for multiclass classification"
            )
    elif objective_fun.find("binary:") == 0:
        if objective_fun in ["binary:logistic", "binary:logitraw"]:
            n_classes = 2
        else:
            raise TypeError(
                "binary:logistic and binary:logitraw are only supported for binary classification"
            )
    else:
        is_regression = True

    if hasattr(booster, "best_iteration"):
        n_iterations = booster.best_iteration + 1
        trees_arr = trees_arr[: n_iterations * (n_classes if n_classes > 2 else 1)]
    else:
        n_iterations = int(len(trees_arr) / (n_classes if n_classes > 2 else 1))

    # Create + base iteration
    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )

    class_label = 0
    iterations_counter = 0
    mis_eq_yes = None
    for tree in trees_arr:
        n_nodes = 1
        # find out the number of nodes in the tree
        for node in tree.split("nodeid")[1:]:
            node_id = int(node[3 : node.find(",")])
            if node_id + 1 > n_nodes:
                n_nodes = node_id + 1
        if is_regression:
            tree_id = mb.create_tree(n_nodes)
        else:
            tree_id = mb.create_tree(n_nodes=n_nodes, class_label=class_label)

        iterations_counter += 1
        if iterations_counter == n_iterations:
            iterations_counter = 0
            class_label += 1
        sub_tree = json.loads(tree)

        # root is leaf
        if "leaf" in sub_tree:
            mb.add_leaf(tree_id=tree_id, response=sub_tree["leaf"], cover=0.0)
            continue

        # add root
        try:
            feature_index = int(sub_tree["split"])
        except ValueError:
            raise TypeError("Feature names must be integers")
        feature_value = np.nextafter(
            np.single(sub_tree["split_condition"]), np.single(-np.inf)
        )
        default_left = int(sub_tree["yes"] == sub_tree["missing"])
        parent_id = mb.add_split(
            tree_id=tree_id,
            feature_index=feature_index,
            feature_value=feature_value,
            default_left=default_left,
            cover=0.0,
        )

        # create queue
        node_queue: Deque[Node] = deque()
        node_queue.append(Node(sub_tree["children"][0], parent_id, 0))
        node_queue.append(Node(sub_tree["children"][1], parent_id, 1))

        # bfs through it
        while node_queue:
            sub_tree = node_queue[0].tree
            parent_id = node_queue[0].parent_id
            position = node_queue[0].position
            node_queue.popleft()

            # current node is leaf
            if "leaf" in sub_tree:
                mb.add_leaf(
                    tree_id=tree_id,
                    response=sub_tree["leaf"],
                    parent_id=parent_id,
                    position=position,
                    cover=0.0,
                )
                continue

            # current node is split
            try:
                feature_index = int(sub_tree["split"])
            except ValueError:
                raise TypeError("Feature names must be integers")
            feature_value = np.nextafter(
                np.single(sub_tree["split_condition"]), np.single(-np.inf)
            )
            default_left = int(sub_tree["yes"] == sub_tree["missing"])

            parent_id = mb.add_split(
                tree_id=tree_id,
                feature_index=feature_index,
                feature_value=feature_value,
                default_left=default_left,
                parent_id=parent_id,
                position=position,
                cover=0.0,
            )

            # append to queue
            node_queue.append(Node(sub_tree["children"][0], parent_id, 0))
            node_queue.append(Node(sub_tree["children"][1], parent_id, 1))

    return mb.model(base_score=base_score)


def get_catboost_params(booster):
    with NamedTemporaryFile() as fp:
        booster.save_model(fp.name, "json")
        fp.seek(0)
        model_data = json.load(fp)
    return model_data


class CatBoostNode:
    def __init__(
        self,
        split: Optional[float] = None,
        value: Optional[List[float]] = None,
        right: Optional[int] = None,
        left: Optional[float] = None,
        cover: Optional[float] = None,
    ) -> None:
        self.split = split
        self.value = value
        self.right = right
        self.left = left
        self.cover = cover


class CatBoostModelData:
    """Wrapper around the CatBoost model dump for easier access to properties"""

    def __init__(self, data):
        self.__data = data

    @property
    def n_features(self):
        return len(self.__data["features_info"]["float_features"])

    @property
    def grow_policy(self):
        return self.__data["model_info"]["params"]["tree_learner_options"]["grow_policy"]

    @property
    def oblivious_trees(self):
        return self.__data["oblivious_trees"]

    @property
    def trees(self):
        return self.__data["trees"]

    @property
    def n_classes(self):
        """Number of classes, returns -1 if it's not a classification model"""
        if "class_params" in self.__data["model_info"]:
            return len(self.__data["model_info"]["class_params"]["class_to_label"])
        return -1

    @property
    def is_classification(self):
        return "class_params" in self.__data

    @property
    def has_categorical_features(self):
        return "categorical_features" in self.__data["features_info"]

    @property
    def is_symmetric_tree(self):
        return self.grow_policy == "SymmetricTree"

    @property
    def float_features(self):
        return self.__data["features_info"]["float_features"]

    @property
    def n_iterations(self):
        if self.is_symmetric_tree:
            return len(self.oblivious_trees)
        else:
            return len(self.trees)

    @property
    def bias(self):
        if self.is_classification:
            return 0
        return self.__data["scale_and_bias"][1][0] / self.n_iterations

    @property
    def scale(self):
        if self.is_classification:
            return 1
        else:
            return self.__data["scale_and_bias"][0]

    @property
    def default_left(self):
        dpo = self.__data["model_info"]["params"]["data_processing_options"]
        nan_mode = dpo["float_features_binarization"]["nan_mode"]
        return int(nan_mode.lower() == "min")


def get_value_as_list(node):
    """Make sure the values are a list"""
    values = node["value"]
    if isinstance(values, (list, tuple)):
        return values
    else:
        return [values]


def get_gbt_model_from_catboost_old(model: Any, model_data=None) -> Any:
    if not model.is_fitted():
        raise RuntimeError("Model should be fitted before exporting to daal4py.")

    if model_data is None:
        model_data = get_catboost_params(model)

    if "categorical_features" in model_data["features_info"]:
        raise NotImplementedError(
            "Categorical features are not supported in daal4py Gradient Boosting Trees"
        )

    n_features = len(model_data["features_info"]["float_features"])

    is_symmetric_tree = (
        model_data["model_info"]["params"]["tree_learner_options"]["grow_policy"]
        == "SymmetricTree"
    )

    if is_symmetric_tree:
        n_iterations = len(model_data["oblivious_trees"])
    else:
        n_iterations = len(model_data["trees"])

    n_classes = 0

    if "class_params" in model_data["model_info"]:
        is_classification = True
        n_classes = len(model_data["model_info"]["class_params"]["class_to_label"])
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )
    else:
        is_classification = False
        mb = gbt_reg_model_builder(n_features, n_iterations)

    splits = []

    # Create splits array (all splits are placed sequentially)
    for feature in model_data["features_info"]["float_features"]:
        if feature["borders"]:
            for feature_border in feature["borders"]:
                splits.append(
                    {"feature_index": feature["feature_index"], "value": feature_border}
                )

    if not is_classification:
        bias = model_data["scale_and_bias"][1][0] / n_iterations
        scale = model_data["scale_and_bias"][0]
    else:
        bias = 0
        scale = 1

    trees_explicit = []
    tree_symmetric = []

    if (
        model_data["model_info"]["params"]["data_processing_options"][
            "float_features_binarization"
        ]["nan_mode"]
        == "Min"
    ):
        default_left = 1
    else:
        default_left = 0

    for tree_num in range(n_iterations):
        if is_symmetric_tree:
            if model_data["oblivious_trees"][tree_num]["splits"] is not None:
                # Tree has more than 1 node
                cur_tree_depth = len(model_data["oblivious_trees"][tree_num]["splits"])
            else:
                cur_tree_depth = 0

            tree_symmetric.append(
                (model_data["oblivious_trees"][tree_num], cur_tree_depth)
            )
        else:
            n_nodes = 1
            # Check if node is a leaf (in case of stump)
            if "split" in model_data["trees"][tree_num]:
                # Get number of trees and splits info via BFS
                # Create queue
                nodes_queue = []
                root_node = CatBoostNode(
                    split=splits[model_data["trees"][tree_num]["split"]["split_index"]]
                )
                nodes_queue.append((model_data["trees"][tree_num], root_node))
                while nodes_queue:
                    cur_node_data, cur_node = nodes_queue.pop(0)
                    if "value" in cur_node_data:
                        if isinstance(cur_node_data["value"], list):
                            cur_node.value = [value for value in cur_node_data["value"]]
                        else:
                            cur_node.value = [cur_node_data["value"] * scale + bias]
                    else:
                        cur_node.split = splits[cur_node_data["split"]["split_index"]]
                        left_node = CatBoostNode()
                        right_node = CatBoostNode()
                        cur_node.left = left_node
                        cur_node.right = right_node
                        nodes_queue.append((cur_node_data["left"], left_node))
                        nodes_queue.append((cur_node_data["right"], right_node))
                        n_nodes += 2
            else:
                root_node = CatBoostNode()
                if is_classification and n_classes > 2:
                    root_node.value = [
                        value * scale for value in model_data["trees"][tree_num]["value"]
                    ]
                else:
                    root_node.value = [
                        model_data["trees"][tree_num]["value"] * scale + bias
                    ]
            trees_explicit.append((root_node, n_nodes))

    tree_id = []
    class_label = 0
    count = 0

    # Only 1 tree for each iteration in case of regression or binary classification
    if not is_classification or n_classes == 2:
        n_tree_each_iter = 1
    else:
        n_tree_each_iter = n_classes

    # Create id for trees (for the right order in modelbuilder)
    for i in range(n_iterations):
        for c in range(n_tree_each_iter):
            if is_symmetric_tree:
                n_nodes = 2 ** (tree_symmetric[i][1] + 1) - 1
            else:
                n_nodes = trees_explicit[i][1]

            if is_classification and n_classes > 2:
                tree_id.append(mb.create_tree(n_nodes, class_label))
                count += 1
                if count == n_iterations:
                    class_label += 1
                    count = 0

            elif is_classification:
                tree_id.append(mb.create_tree(n_nodes, 0))
            else:
                tree_id.append(mb.create_tree(n_nodes))

    if is_symmetric_tree:
        for class_label in range(n_tree_each_iter):
            for i in range(n_iterations):
                cur_tree_info = tree_symmetric[i][0]
                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                cur_tree_leaf_val = cur_tree_info["leaf_values"]
                cur_tree_depth = tree_symmetric[i][1]

                if cur_tree_depth == 0:
                    mb.add_leaf(tree_id=cur_tree_id, response=cur_tree_leaf_val[0])
                else:
                    # One split used for the whole level
                    cur_level_split = splits[
                        cur_tree_info["splits"][cur_tree_depth - 1]["split_index"]
                    ]
                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=cur_level_split["feature_index"],
                        feature_value=cur_level_split["value"],
                        default_left=default_left,
                        cover=0.0,
                    )
                    prev_level_nodes = [root_id]

                    # Iterate over levels, splits in json are reversed (root split is the last)
                    for cur_level in range(cur_tree_depth - 2, -1, -1):
                        cur_level_nodes = []
                        for cur_parent in prev_level_nodes:
                            cur_level_split = splits[
                                cur_tree_info["splits"][cur_level]["split_index"]
                            ]
                            cur_left_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=0,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            cur_right_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=1,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            cur_level_nodes.append(cur_left_node)
                            cur_level_nodes.append(cur_right_node)
                        prev_level_nodes = cur_level_nodes

                    # Different storing format for leaves
                    if not is_classification or n_classes == 2:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num]
                                * scale
                                + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=0.0,
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num + 1]
                                * scale
                                + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=0.0,
                            )
                    else:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            left_index = (
                                2 * last_level_node_num * n_tree_each_iter + class_label
                            )
                            right_index = (
                                2 * last_level_node_num + 1
                            ) * n_tree_each_iter + class_label
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[left_index] * scale + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=0.0,
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[right_index] * scale + bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=0.0,
                            )
    else:
        for class_label in range(n_tree_each_iter):
            for i in range(n_iterations):
                root_node = trees_explicit[i][0]

                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                # Traverse tree via BFS and build tree with modelbuilder
                if root_node.value is None:
                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=root_node.split["feature_index"],
                        feature_value=root_node.split["value"],
                        default_left=default_left,
                        cover=0.0,
                    )
                    nodes_queue = [(root_node, root_id)]
                    while nodes_queue:
                        cur_node, cur_node_id = nodes_queue.pop(0)
                        left_node = cur_node.left
                        # Check if node is a leaf
                        if left_node.value is None:
                            left_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=0,
                                feature_index=left_node.split["feature_index"],
                                feature_value=left_node.split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((left_node, left_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=left_node.value[class_label],
                                parent_id=cur_node_id,
                                position=0,
                                cover=0.0,
                            )
                        right_node = cur_node.right
                        # Check if node is a leaf
                        if right_node.value is None:
                            right_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=1,
                                feature_index=right_node.split["feature_index"],
                                feature_value=right_node.split["value"],
                                default_left=default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((right_node, right_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_node.right.value[class_label],
                                parent_id=cur_node_id,
                                position=1,
                                cover=0.0,
                            )

                else:
                    # Tree has only one node
                    mb.add_leaf(
                        tree_id=cur_tree_id,
                        response=root_node.value[class_label],
                        cover=0.0,
                    )

    warn("Models converted from CatBoost cannot be used for SHAP value calculation")
    return mb.model(0.0)


def calc_node_weights_from_leaf_weights(weights):
    def sum_pairs(values):
        assert len(values) % 2 == 0, "Length of values must be even"
        return [values[i] + values[i + 1] for i in range(0, len(values), 2)]

    level_weights = sum_pairs(weights)
    result = [level_weights]
    while len(level_weights) > 1:
        level_weights = sum_pairs(level_weights)
        result.append(level_weights)
    return result[::-1]


def get_gbt_model_from_catboost(booster: Any) -> Any:
    if not booster.is_fitted():
        raise RuntimeError("Model should be fitted before exporting to daal4py.")

    model = CatBoostModelData(get_catboost_params(booster))

    if model.has_categorical_features:
        raise NotImplementedError(
            "Categorical features are not supported in daal4py Gradient Boosting Trees"
        )

    if model.is_classification:
        mb = gbt_clf_model_builder(
            n_features=model.n_features,
            n_iterations=model.n_iterations,
            n_classes=model.n_classes,
        )
    else:
        mb = gbt_reg_model_builder(
            n_features=model.n_features, n_iterations=model.n_iterations
        )

    # Create splits array (all splits are placed sequentially)
    splits = []
    for feature in model.float_features:
        if feature["borders"]:
            for feature_border in feature["borders"]:
                splits.append(
                    {"feature_index": feature["feature_index"], "value": feature_border}
                )

    trees_explicit = []
    tree_symmetric = []

    if model.is_symmetric_tree:
        for tree in model.oblivious_trees:
            cur_tree_depth = len(tree.get("splits", []))
            tree_symmetric.append((tree, cur_tree_depth))
    else:
        for tree in model.trees:
            n_nodes = 1
            if "split" not in tree:
                # handle leaf node
                values = get_value_as_list(tree)
                root_node = CatBoostNode(value=[value * model.scale for value in values])
                continue
            # Check if node is a leaf (in case of stump)
            if "split" in tree:
                # Get number of trees and splits info via BFS
                # Create queue
                nodes_queue = []
                root_node = CatBoostNode(split=splits[tree["split"]["split_index"]])
                nodes_queue.append((tree, root_node))
                while nodes_queue:
                    cur_node_data, cur_node = nodes_queue.pop(0)
                    if "value" in cur_node_data:
                        cur_node.value = get_value_as_list(cur_node_data)
                    else:
                        cur_node.split = splits[cur_node_data["split"]["split_index"]]
                        left_node = CatBoostNode()
                        right_node = CatBoostNode()
                        cur_node.left = left_node
                        cur_node.right = right_node
                        nodes_queue.append((cur_node_data["left"], left_node))
                        nodes_queue.append((cur_node_data["right"], right_node))
                        n_nodes += 2
            else:
                root_node = CatBoostNode()
                if model.is_classification and model.n_classes > 2:
                    root_node.value = [value * model.scale for value in tree["value"]]
                else:
                    root_node.value = [tree["value"] * model.scale + model.bias]
            trees_explicit.append((root_node, n_nodes))

    tree_id = []
    class_label = 0
    count = 0

    # Only 1 tree for each iteration in case of regression or binary classification
    if not model.is_classification or model.n_classes == 2:
        n_tree_each_iter = 1
    else:
        n_tree_each_iter = model.n_classes

    shap_ready = False

    # Create id for trees (for the right order in model builder)
    for i in range(model.n_iterations):
        for _ in range(n_tree_each_iter):
            if model.is_symmetric_tree:
                n_nodes = 2 ** (tree_symmetric[i][1] + 1) - 1
            else:
                n_nodes = trees_explicit[i][1]

            if model.is_classification and model.n_classes > 2:
                tree_id.append(mb.create_tree(n_nodes, class_label))
                count += 1
                if count == model.n_iterations:
                    class_label += 1
                    count = 0

            elif model.is_classification:
                tree_id.append(mb.create_tree(n_nodes, 0))
            else:
                tree_id.append(mb.create_tree(n_nodes))

    if model.is_symmetric_tree:
        for class_label in range(n_tree_each_iter):
            for i in range(model.n_iterations):
                shap_ready = True  # this code branch provides all info for SHAP values
                cur_tree_info = tree_symmetric[i][0]
                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                cur_tree_leaf_val = cur_tree_info["leaf_values"]
                cur_tree_leaf_weights = cur_tree_info["leaf_weights"]
                cur_tree_depth = tree_symmetric[i][1]
                if cur_tree_depth == 0:
                    mb.add_leaf(
                        tree_id=cur_tree_id,
                        response=cur_tree_leaf_val[0],
                        cover=cur_tree_leaf_weights[0],
                    )
                else:
                    # One split used for the whole level
                    cur_level_split = splits[
                        cur_tree_info["splits"][cur_tree_depth - 1]["split_index"]
                    ]
                    cur_tree_weights_per_level = calc_node_weights_from_leaf_weights(
                        cur_tree_leaf_weights
                    )
                    root_weight = cur_tree_weights_per_level[0][0]
                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=cur_level_split["feature_index"],
                        feature_value=cur_level_split["value"],
                        default_left=model.default_left,
                        cover=root_weight,
                    )
                    prev_level_nodes = [root_id]

                    # Iterate over levels, splits in json are reversed (root split is the last)
                    for cur_level in range(cur_tree_depth - 2, -1, -1):
                        cur_level_nodes = []
                        next_level_weights = cur_tree_weights_per_level[cur_level + 1]
                        cur_level_node_index = 0
                        for cur_parent in prev_level_nodes:
                            cur_level_split = splits[
                                cur_tree_info["splits"][cur_level]["split_index"]
                            ]
                            cur_left_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=0,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=model.default_left,
                                cover=next_level_weights[cur_level_node_index],
                            )
                            # cur_level_node_index += 1
                            cur_right_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=1,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=model.default_left,
                                cover=next_level_weights[cur_level_node_index],
                            )
                            # cur_level_node_index += 1
                            cur_level_nodes.append(cur_left_node)
                            cur_level_nodes.append(cur_right_node)
                        prev_level_nodes = cur_level_nodes

                    # Different storing format for leaves
                    if not model.is_classification or model.n_classes == 2:
                        shap_ready = False
                        for last_level_node_num in range(len(prev_level_nodes)):
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num]
                                * model.scale
                                + model.bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=0.0,
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num + 1]
                                * model.scale
                                + model.bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=0.0,
                            )
                    else:
                        shap_ready = False
                        for last_level_node_num in range(len(prev_level_nodes)):
                            left_index = (
                                2 * last_level_node_num * n_tree_each_iter + class_label
                            )
                            right_index = (
                                2 * last_level_node_num + 1
                            ) * n_tree_each_iter + class_label
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[left_index] * model.scale
                                + model.bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=0.0,
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[right_index] * model.scale
                                + model.bias,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=0.0,
                            )
    else:
        shap_ready = False
        for class_label in range(n_tree_each_iter):
            for i in range(model.n_iterations):
                root_node = trees_explicit[i][0]

                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                # Traverse tree via BFS and build tree with modelbuilder
                if root_node.value is None:
                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=root_node.split["feature_index"],
                        feature_value=root_node.split["value"],
                        default_left=model.default_left,
                        cover=0.0,
                    )
                    nodes_queue = [(root_node, root_id)]
                    while nodes_queue:
                        cur_node, cur_node_id = nodes_queue.pop(0)
                        left_node = cur_node.left
                        # Check if node is a leaf
                        if left_node.value is None:
                            left_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=0,
                                feature_index=left_node.split["feature_index"],
                                feature_value=left_node.split["value"],
                                default_left=model.default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((left_node, left_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=left_node.value[class_label],
                                parent_id=cur_node_id,
                                position=0,
                                cover=0.0,
                            )
                        right_node = cur_node.right
                        # Check if node is a leaf
                        if right_node.value is None:
                            right_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=1,
                                feature_index=right_node.split["feature_index"],
                                feature_value=right_node.split["value"],
                                default_left=model.default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((right_node, right_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_node.right.value[class_label],
                                parent_id=cur_node_id,
                                position=1,
                                cover=0.0,
                            )

                else:
                    # Tree has only one node
                    mb.add_leaf(
                        tree_id=cur_tree_id,
                        response=root_node.value[class_label],
                        cover=0.0,
                    )

    if not shap_ready:
        warn("Converted models of this type do not support SHAP value calculation")
    return mb.model(base_score=0.0)


def test_xgb_early_stopping():
    num_classes = 3
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_classes=num_classes,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = X[:500, :], X[500:, :], y[:500], y[500:]

    # training parameters setting
    params = {
        "n_estimators": 100,
        "max_bin": 256,
        "scale_pos_weight": 2,
        "lambda_l2": 1,
        "alpha": 0.9,
        "max_depth": 8,
        "num_leaves": 2**8,
        "verbosity": 0,
        "objective": "multi:softproba",
        "learning_rate": 0.3,
        "num_class": num_classes,
        "early_stopping_rounds": 5,
        "verbose_eval": False,
    }

    xgb_clf = xgb.XGBClassifier(**params)
    xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    get_gbt_model_from_xgboost(xgb_clf.get_booster())


def test_xgb_regressor_model_dump_size():
    X, y = make_regression(random_state=3, n_samples=2000)

    for depth in range(2, 18):
        xgb_model = xgb.XGBRegressor(max_depth=depth, n_estimators=10, random_state=3)
        print(f"{X.shape=}")
        print("Running fit...")
        xgb_model.fit(X, y)

        booster = xgb_model.get_booster()
        booster.feature_names = [str(i) for i in range(booster.num_features())]

        tree_dupes = 40
        orig_dump = booster.get_dump(dump_format="json", with_stats=True)

        def new_dump(*args, **kwargs):
            return orig_dump * tree_dupes

        booster.get_dump = new_dump
        booster.best_iteration = tree_dupes * len(orig_dump) - 1

        start = time.time()
        d4p_model = get_gbt_model_from_xgboost(booster)
        print(f"maxDepth = {depth}, model created in {time.time() - start:.2f} s")

        with open(f"test-new-maxDepth{depth}.pkl", "wb") as fp:
            pickle.dump(d4p_model, fp)

    # print("Start model building (old)")
    # start = time.time()
    # d4p_model = get_gbt_model_from_xgboost_old(booster)
    # print(f"Model created in {time.time() - start:.2f} s")

    # with open("test-old.pkl", "wb") as fp:
    #     pickle.dump(d4p_model, fp)


def test_xgb_regressor():
    print("Gen data...")
    X, y = make_regression(random_state=3, n_samples=2000)

    xgb_model = xgb.XGBRegressor(max_depth=16, n_estimators=10, random_state=3)
    print(f"{X.shape=}")
    print("Running fit...")
    xgb_model.fit(X, y)
    print("Start model building")
    start = time.time()
    d4p_model = get_gbt_model_from_xgboost(xgb_model.get_booster())
    print(f"Model created in {time.time() - start:.2f} s")


def test_xgb_classification():
    X, y = make_classification(random_state=3)

    xgb_model = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=50,
        random_state=3,
        base_score=0.7,
        objective="binary:logitraw",
    )
    print(f"{X.shape=}")
    print("Running fit...")
    xgb_model.fit(X, y)

    booster = xgb_model.get_booster()

    print("Start model building")
    start = time.time()
    d4p_model = get_gbt_model_from_xgboost(booster)

    print(f"Model created in {time.time() - start:.2f} s")
    # np.testing.assert_allclose(xgb_model.predict_proba(X[:2,:]), d4p_model.predict_proba(X[:2,:]), rtol=1e-6)


def test_lightgbm_regressor():
    print("Gen data...")
    X, y = make_regression(n_samples=50, random_state=42, n_features=10)

    print(f"{X.shape=}")
    print("Running fit...")

    params = {
        "task": "train",
        "boosting": "gbdt",
        "objective": "regression",
        "num_leaves": 10,
        "learning_rage": 0.05,
        "metric": {"l2", "l1"},
        "verbose": -1,
    }

    train = lgbm.Dataset(X, y)
    lgbm_model = lgbm.train(params, train_set=train)

    print("Start model building")
    start = time.time()
    d4p_model = get_gbt_model_from_lightgbm(lgbm_model)
    print(f"Model created in {time.time() - start:.2f} s")


def test_lightgbm_classification():
    print("Gen data...")
    X, y = make_classification(random_state=3, n_classes=3, n_informative=3)
    print(f"{X.shape=}")
    print("Running fit...")

    params = {
        "n_estimators": 1,
        "task": "train",
        "boosting": "gbdt",
        "objective": "multiclass",
        "num_leaves": 4,
        "num_class": 3,
    }

    train = lgbm.Dataset(X, y)
    lgbm_model = lgbm.train(params, train_set=train)

    print("Start model building")
    start = time.time()
    d4p_model = get_gbt_model_from_lightgbm(lgbm_model)
    print(f"Model created in {time.time() - start:.2f} s")


def test_catboost_regressor():
    print("Gen data...")
    X, y = make_regression(n_samples=2000, random_state=3)

    print(f"{X.shape=}")
    print("Running fit...")

    model = cb.CatBoostRegressor(max_depth=5, n_estimators=50)
    model.fit(X, y)

    print("Start model building")
    # overload get_gbt_model_from_catboost
    daal4py.get_gbt_model_from_catboost = get_gbt_model_from_catboost
    start = time.time()
    d4p_model = daal4py.mb.convert_model(model)
    print(f"Model created in {time.time() - start:.2f} s")

    d4p_pred = d4p_model.predict(X[:2])
    pred = model.predict(X[:2])
    np.testing.assert_allclose(d4p_pred, pred)


def test_catboost_classification():
    print("Gen data...")
    X, y = make_classification(random_state=3, n_classes=3, n_informative=3)

    print(f"{X.shape=}")
    print("Running fit...")

    model = cb.CatBoostClassifier(max_depth=3, n_estimators=5, classes_count=3)
    model.fit(X, y)

    print("Start model building")
    start = time.time()
    d4p_model = get_gbt_model_from_catboost(model)
    print(f"Model created in {time.time() - start:.2f} s")


if __name__ == "__main__":
    # test_xgb_early_stopping()
    # test_xgb_regressor()
    # test_xgb_regressor_model_dump_size()
    # test_xgb_classification()
    # test_lightgbm_regressor()
    # test_lightgbm_classification()
    test_catboost_regressor()
    # test_catboost_classification()
