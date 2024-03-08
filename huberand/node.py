import json
import multiprocessing
import threading
import time
from collections import deque
from tempfile import NamedTemporaryFile
from typing import Any, Deque, Dict, Generator, List, Optional, Tuple
from warnings import warn

import catboost as cb
import lightgbm as lgbm
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

from daal4py import gbt_clf_model_builder, gbt_reg_model_builder


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


def _add_tree_to_model(tree: TreeView, mb: Any):
    """Wrapper to add one tree to an existing model (built with ModelBuilder `mb`)"""
    if tree.is_leaf:
        mb.add_leaf(tree_id=tree.tree_id, response=tree.value, cover=tree.cover)
        return

    root_node = tree.root_node
    parent_id = mb.add_split(
        tree_id=tree.tree_id,
        feature_index=root_node.feature,
        feature_value=root_node.value,
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
        _add_tree_to_model(tree, mb)

    return mb.model(base_score=base_score)
