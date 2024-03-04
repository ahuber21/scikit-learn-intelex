# ==============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import os
import sys
import types
from functools import lru_cache
from typing import Any, Dict, Optional

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

logger = logging.getLogger("sklearnex")


def _is_new_patching_available():
    return os.environ.get("OFF_ONEDAL_IFACE") is None and daal_check_version(
        (2021, "P", 300)
    )


def _is_preview_enabled():
    return os.environ.get("SKLEARNEX_PREVIEW") is not None


@lru_cache(maxsize=None)
def get_patch_map_core(preview=False):
    if preview:
        # use recursion to guarantee that state of preview
        # and non-preview maps are done at the same time.
        # The two lru_cache dicts are actually one underneath.
        # Preview is always secondary. Both sklearnex patch
        # maps are referring to the daal4py dict unless the
        # key has been replaced. Use with caution.
        mapping = get_patch_map_core().copy()

        if _is_new_patching_available():
            import sklearn.covariance as covariance_module

            # Preview classes for patching
            from .preview.cluster import KMeans as KMeans_sklearnex
            from .preview.covariance import (
                EmpiricalCovariance as EmpiricalCovariance_sklearnex,
            )
            from .preview.decomposition import PCA as PCA_sklearnex

            # Since the state of the lru_cache without preview cannot be
            # guaranteed to not have already enabled sklearnex algorithms
            # when preview is used, setting the mapping element[1] to None
            # should NOT be done. This may lose track of the unpatched
            # sklearn estimator or function.
            # PCA
            decomposition_module, _, _ = mapping["pca"][0][0]
            sklearn_obj = mapping["pca"][0][1]
            mapping.pop("pca")
            mapping["pca"] = [[(decomposition_module, "PCA", PCA_sklearnex), sklearn_obj]]

            # KMeans
            cluster_module, _, _ = mapping["kmeans"][0][0]
            sklearn_obj = mapping["kmeans"][0][1]
            mapping.pop("kmeans")
            mapping["kmeans"] = [
                [(cluster_module, "kmeans", KMeans_sklearnex), sklearn_obj]
            ]

            # Covariance
            mapping["empiricalcovariance"] = [
                [
                    (
                        covariance_module,
                        "EmpiricalCovariance",
                        EmpiricalCovariance_sklearnex,
                    ),
                    None,
                ]
            ]
        return mapping

    from daal4py.sklearn.monkeypatch.dispatcher import _get_map_of_algorithms

    # NOTE: this is a shallow copy of a dict, modification is dangerous
    mapping = _get_map_of_algorithms().copy()

    # NOTE: Use of daal4py _get_map_of_algorithms and
    # get_patch_map/get_patch_map_core should not be used concurrently.
    # The setting of elements to None below may cause loss of state
    # when interacting with sklearn. A dictionary key must not be
    # modified but totally replaced, otherwise it will cause chaos.
    # Hence why pop is being used.
    if _is_new_patching_available():
        # Scikit-learn* modules
        import sklearn as base_module
        import sklearn.cluster as cluster_module
        import sklearn.decomposition as decomposition_module
        import sklearn.ensemble as ensemble_module
        import sklearn.linear_model as linear_model_module
        import sklearn.neighbors as neighbors_module
        import sklearn.svm as svm_module

        if sklearn_check_version("1.2.1"):
            import sklearn.utils.parallel as parallel_module
        else:
            import sklearn.utils.fixes as parallel_module

        # Classes and functions for patching
        from ._config import config_context as config_context_sklearnex
        from ._config import get_config as get_config_sklearnex
        from ._config import set_config as set_config_sklearnex

        if sklearn_check_version("1.2.1"):
            from .utils.parallel import _FuncWrapper as _FuncWrapper_sklearnex
        else:
            from .utils.parallel import _FuncWrapperOld as _FuncWrapper_sklearnex

        from .cluster import DBSCAN as DBSCAN_sklearnex
        from .ensemble import ExtraTreesClassifier as ExtraTreesClassifier_sklearnex
        from .ensemble import ExtraTreesRegressor as ExtraTreesRegressor_sklearnex
        from .ensemble import RandomForestClassifier as RandomForestClassifier_sklearnex
        from .ensemble import RandomForestRegressor as RandomForestRegressor_sklearnex
        from .linear_model import LinearRegression as LinearRegression_sklearnex
        from .linear_model import LogisticRegression as LogisticRegression_sklearnex
        from .neighbors import KNeighborsClassifier as KNeighborsClassifier_sklearnex
        from .neighbors import KNeighborsRegressor as KNeighborsRegressor_sklearnex
        from .neighbors import LocalOutlierFactor as LocalOutlierFactor_sklearnex
        from .neighbors import NearestNeighbors as NearestNeighbors_sklearnex
        from .svm import SVC as SVC_sklearnex
        from .svm import SVR as SVR_sklearnex
        from .svm import NuSVC as NuSVC_sklearnex
        from .svm import NuSVR as NuSVR_sklearnex

        # DBSCAN
        mapping.pop("dbscan")
        mapping["dbscan"] = [[(cluster_module, "DBSCAN", DBSCAN_sklearnex), None]]

        # SVM
        mapping.pop("svm")
        mapping.pop("svc")
        mapping["svr"] = [[(svm_module, "SVR", SVR_sklearnex), None]]
        mapping["svc"] = [[(svm_module, "SVC", SVC_sklearnex), None]]
        mapping["nusvr"] = [[(svm_module, "NuSVR", NuSVR_sklearnex), None]]
        mapping["nusvc"] = [[(svm_module, "NuSVC", NuSVC_sklearnex), None]]

        # Linear Regression
        mapping.pop("linear")
        mapping.pop("linearregression")
        mapping["linear"] = [
            [
                (
                    linear_model_module,
                    "LinearRegression",
                    LinearRegression_sklearnex,
                ),
                None,
            ]
        ]
        mapping["linearregression"] = mapping["linear"]

        # Logistic Regression

        mapping.pop("logisticregression")
        mapping.pop("log_reg")
        mapping.pop("logistic")
        mapping.pop("_logistic_regression_path")
        mapping["log_reg"] = [
            [
                (
                    linear_model_module,
                    "LogisticRegression",
                    LogisticRegression_sklearnex,
                ),
                None,
            ]
        ]
        mapping["logisticregression"] = mapping["log_reg"]

        # kNN
        mapping.pop("knn_classifier")
        mapping.pop("kneighborsclassifier")
        mapping.pop("knn_regressor")
        mapping.pop("kneighborsregressor")
        mapping.pop("nearest_neighbors")
        mapping.pop("nearestneighbors")
        mapping["knn_classifier"] = [
            [
                (
                    neighbors_module,
                    "KNeighborsClassifier",
                    KNeighborsClassifier_sklearnex,
                ),
                None,
            ]
        ]
        mapping["knn_regressor"] = [
            [
                (
                    neighbors_module,
                    "KNeighborsRegressor",
                    KNeighborsRegressor_sklearnex,
                ),
                None,
            ]
        ]
        mapping["nearest_neighbors"] = [
            [(neighbors_module, "NearestNeighbors", NearestNeighbors_sklearnex), None]
        ]
        mapping["kneighborsclassifier"] = mapping["knn_classifier"]
        mapping["kneighborsregressor"] = mapping["knn_regressor"]
        mapping["nearestneighbors"] = mapping["nearest_neighbors"]

        # Ensemble
        mapping["extra_trees_classifier"] = [
            [
                (
                    ensemble_module,
                    "ExtraTreesClassifier",
                    ExtraTreesClassifier_sklearnex,
                ),
                None,
            ]
        ]
        mapping["extra_trees_regressor"] = [
            [
                (
                    ensemble_module,
                    "ExtraTreesRegressor",
                    ExtraTreesRegressor_sklearnex,
                ),
                None,
            ]
        ]
        mapping["extratreesclassifier"] = mapping["extra_trees_classifier"]
        mapping["extratreesregressor"] = mapping["extra_trees_regressor"]
        mapping.pop("random_forest_classifier")
        mapping.pop("random_forest_regressor")
        mapping.pop("randomforestclassifier")
        mapping.pop("randomforestregressor")
        mapping["random_forest_classifier"] = [
            [
                (
                    ensemble_module,
                    "RandomForestClassifier",
                    RandomForestClassifier_sklearnex,
                ),
                None,
            ]
        ]
        mapping["random_forest_regressor"] = [
            [
                (
                    ensemble_module,
                    "RandomForestRegressor",
                    RandomForestRegressor_sklearnex,
                ),
                None,
            ]
        ]
        mapping["randomforestclassifier"] = mapping["random_forest_classifier"]
        mapping["randomforestregressor"] = mapping["random_forest_regressor"]

        # LocalOutlierFactor
        mapping["lof"] = [
            [
                (neighbors_module, "LocalOutlierFactor", LocalOutlierFactor_sklearnex),
                None,
            ]
        ]
        mapping["localoutlierfactor"] = mapping["lof"]

        # Configs
        mapping["set_config"] = [
            [(base_module, "set_config", set_config_sklearnex), None]
        ]
        mapping["get_config"] = [
            [(base_module, "get_config", get_config_sklearnex), None]
        ]
        mapping["config_context"] = [
            [(base_module, "config_context", config_context_sklearnex), None]
        ]

        # Necessary for proper work with multiple threads
        mapping["parallel.get_config"] = [
            [(parallel_module, "get_config", get_config_sklearnex), None]
        ]
        mapping["_funcwrapper"] = [
            [(parallel_module, "_FuncWrapper", _FuncWrapper_sklearnex), None]
        ]
    return mapping


# This is necessary to properly cache the patch_map when
# using preview.
def get_patch_map():
    preview = _is_preview_enabled()
    return get_patch_map_core(preview=preview)


get_patch_map.cache_clear = get_patch_map_core.cache_clear


get_patch_map.cache_info = get_patch_map_core.cache_info


def get_patch_names():
    return list(get_patch_map().keys())


def check_entity_loaded(
    name: Optional[str] = None, modules: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    This function checks if a specified module or class is already loaded in sys.modules.

    Parameters:
    name (str, optional): The name of the module or class to check. If no name is provided, the function checks for 'sklearn'.

    Returns:
    str: A warning message if the specified module or class is already loaded, indicating that patch_sklearn()
    should be called before any import statements from that module or class. If the module or class is not loaded,
    the function returns None.

    Example:
    >>> check_entity_loaded()
    'sklearn or some parts of it are already loaded. patch_sklearn() only affects classes imported *after* calling it.
    To retrieve patched entities, make sure to call patch_sklearn() before any import statements from sklearn.'

    >>> check_entity_loaded(name='LogisticRegression')
    'LogisticRegression or some parts of it are already loaded. patch_sklearn() only affects classes imported *after*
    calling it. To retrieve patched entities, make sure to call patch_sklearn() before any import statements from LogisticRegression.'
    """

    def _get_loaded_classes():
        loaded_classes = []
        for key, module in sys.modules.items():
            if "sklearn" in key and isinstance(module, types.ModuleType):
                loaded_classes.extend(
                    [
                        cls.__name__
                        for cls in vars(module).values()
                        if isinstance(cls, type)
                    ]
                )
        return loaded_classes

    # list of all loaded modules, uses sys.modules per default
    modules = modules if modules is not None else sys.modules

    # is `name` or anything from sklearn already loaded?
    loaded = name is None and "sklearn" in modules.keys()
    loaded |= name is not None and name in _get_loaded_classes()

    if loaded:
        return (
            f"{name or 'sklearn'} or some parts of it are already loaded. "
            "patch_sklearn() only affects classes imported *after* calling it. "
            "To retrieve patched entities, make sure to call patch_sklearn() before any import statements from sklearn."
        )


def patch_sklearn(
    name=None, verbose=True, global_patch=False, preview=False, no_msg=False
):
    if preview:
        os.environ["SKLEARNEX_PREVIEW"] = "enabled_via_patch_sklearn"
    if not sklearn_check_version("0.22"):
        raise NotImplementedError(
            "Intel(R) Extension for Scikit-learn* patches apply "
            "for scikit-learn >= 0.22 only ..."
        )

    if not no_msg and (msg := check_entity_loaded(name)) is not None:
        logger.warning(msg)

    if global_patch:
        from sklearnex.glob.dispatcher import patch_sklearn_global

        patch_sklearn_global(name, verbose)

    from daal4py.sklearn import patch_sklearn as patch_sklearn_orig

    if _is_new_patching_available():
        for config in ["set_config", "get_config", "config_context"]:
            patch_sklearn_orig(
                config, verbose=False, deprecation=False, get_map=get_patch_map
            )
    if isinstance(name, list):
        for algorithm in name:
            patch_sklearn_orig(
                algorithm, verbose=False, deprecation=False, get_map=get_patch_map
            )
    else:
        patch_sklearn_orig(name, verbose=False, deprecation=False, get_map=get_patch_map)

    if verbose and sys.stderr is not None:
        sys.stderr.write(
            "Intel(R) Extension for Scikit-learn* enabled "
            "(https://github.com/intel/scikit-learn-intelex)\n"
        )


def unpatch_sklearn(name=None, global_unpatch=False):
    if global_unpatch:
        from sklearnex.glob.dispatcher import unpatch_sklearn_global

        unpatch_sklearn_global()
    from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig

    if isinstance(name, list):
        for algorithm in name:
            unpatch_sklearn_orig(algorithm, get_map=get_patch_map)
    else:
        if _is_new_patching_available():
            for config in ["set_config", "get_config", "config_context"]:
                unpatch_sklearn_orig(config, get_map=get_patch_map)
        unpatch_sklearn_orig(name, get_map=get_patch_map)
    if os.environ.get("SKLEARNEX_PREVIEW") == "enabled_via_patch_sklearn":
        os.environ.pop("SKLEARNEX_PREVIEW")


def sklearn_is_patched(name=None, return_map=False):
    from daal4py.sklearn import sklearn_is_patched as sklearn_is_patched_orig

    if isinstance(name, list):
        if return_map:
            result = {}
            for algorithm in name:
                result[algorithm] = sklearn_is_patched_orig(
                    algorithm, get_map=get_patch_map
                )
            return result
        else:
            is_patched = True
            for algorithm in name:
                is_patched = is_patched and sklearn_is_patched_orig(
                    algorithm, get_map=get_patch_map
                )
            return is_patched
    else:
        return sklearn_is_patched_orig(name, get_map=get_patch_map, return_map=return_map)


def is_patched_instance(instance: object) -> bool:
    """Returns True if the `instance` is patched with scikit-learn-intelex"""
    module = getattr(instance, "__module__", "")
    return ("daal4py" in module) or ("sklearnex" in module)
