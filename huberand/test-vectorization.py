import cProfile

# import pandas as pd
import os
import sys

import numpy as np

np.random.seed(42)


def generate_data(size):
    if size == "large":
        X = np.random.rand(1000000, 3000)
        n_clusters = 1000
        return (X, n_clusters)
    elif size == "medium":
        X = np.random.rand(100000, 1000)
        n_clusters = 500
        return (X, n_clusters)
    else:
        X = np.random.rand(100000, 100)
        n_clusters = 100
        return (X, n_clusters)


size = "large"
X, n_clusters = generate_data(size)
is_preview = True
if is_preview:
    os.environ["SKLEARNEX_PREVIEW"] = "1"
    print("######### Preview Enabled ###########")
else:
    print("######### Master Enabled ###########")

from sklearnex import patch_sklearn

patch_sklearn()
import sklearn

sklearn.set_config(assume_finite=True)
from sklearn.cluster import KMeans


def profile_fit(n_clusters, n_iterations, init_method):
    for i in range(n_iterations):
        # print(i)
        kmeans = KMeans(
            n_clusters=n_clusters, max_iter=10, init=init_method, n_init=1
        ).fit(X)


def profile_predict(model, n_iterations):
    for i in range(n_iterations):
        # print(i)
        X_tr = model.predict(X)


is_fit = False
n_iterations = 25
init_method = "k-means++"
filename = "kmeans_preview_fit_kmeanspp.prof"


input(f"Ready - PID {os.getpid()}")
if is_fit:
    # cProfile.run("profile_fit(n_clusters, n_iterations, init_method)", filename)
    profile_fit(n_clusters, n_iterations, init_method)
else:
    kmeans = KMeans(n_clusters=n_clusters, max_iter=10, init=init_method, n_init=1).fit(X)
    # cProfile.run("profile_predict(kmeans, n_iterations)", filename)
    profile_predict(kmeans, n_iterations)
