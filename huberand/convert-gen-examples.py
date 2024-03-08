from types import NoneType

gen_examples = [
    ("adaboost", None, None, (2020, "P", 0)),
    ("adagrad_mse", "adagrad_mse.csv", "minimum"),
    ("association_rules", "association_rules.csv", "confidence"),
    ("bacon_outlier", "multivariate_outlier.csv", lambda r: r[1].weights),
    ("brownboost", None, None, (2020, "P", 0)),
    (
        "correlation_distance",
        "correlation_distance.csv",
        lambda r: [
            [np.amin(r.correlationDistance)],
            [np.amax(r.correlationDistance)],
            [np.mean(r.correlationDistance)],
            [np.average(r.correlationDistance)],
        ],
    ),
    (
        "cosine_distance",
        "cosine_distance.csv",
        lambda r: [
            [np.amin(r.cosineDistance)],
            [np.amax(r.cosineDistance)],
            [np.mean(r.cosineDistance)],
            [np.average(r.cosineDistance)],
        ],
    ),
    # ('gradient_boosted_regression', 'gradient_boosted_regression.csv',
    #  lambda x: x[1].prediction),
    ("cholesky", "cholesky.csv", "choleskyFactor"),
    ("covariance", "covariance.csv", "covariance"),
    ("covariance_streaming", "covariance.csv", "covariance"),
    (
        "decision_forest_classification_default_dense",
        None,
        lambda r: r[1].prediction,
        (2023, "P", 1),
    ),
    (
        "decision_forest_classification_hist",
        None,
        lambda r: r[1].prediction,
        (2023, "P", 1),
    ),
    (
        "decision_forest_regression_default_dense",
        "decision_forest_regression.csv",
        lambda r: r[1].prediction,
        (2023, "P", 1),
    ),
    (
        "decision_forest_regression_hist",
        "decision_forest_regression.csv",
        lambda r: r[1].prediction,
        (2023, "P", 1),
    ),
    (
        "decision_forest_regression_default_dense",
        "decision_forest_regression_20230101.csv",
        lambda r: r[1].prediction,
        (2023, "P", 101),
    ),
    (
        "decision_forest_regression_hist",
        "decision_forest_regression_20230101.csv",
        lambda r: r[1].prediction,
        (2023, "P", 101),
    ),
    (
        "decision_tree_classification",
        "decision_tree_classification.csv",
        lambda r: r[1].prediction,
    ),
    (
        "decision_tree_regression",
        "decision_tree_regression.csv",
        lambda r: r[1].prediction,
    ),
    ("distributions_bernoulli",),
    ("distributions_normal",),
    ("distributions_uniform",),
    ("em_gmm", "em_gmm.csv", lambda r: r.covariances[0]),
    ("gradient_boosted_classification",),
    ("gradient_boosted_regression",),
    ("implicit_als", "implicit_als.csv", "prediction"),
    ("kdtree_knn_classification", None, None),
    ("kmeans", "kmeans.csv", "centroids"),
    ("lbfgs_cr_entr_loss", "lbfgs_cr_entr_loss.csv", "minimum"),
    ("lbfgs_mse", "lbfgs_mse.csv", "minimum"),
    ("linear_regression", "linear_regression.csv", lambda r: r[1].prediction),
    ("linear_regression_streaming", "linear_regression.csv", lambda r: r[1].prediction),
    ("log_reg_binary_dense", "log_reg_binary_dense.csv", lambda r: r[1].prediction),
    ("log_reg_binary_dense", None, None),
    ("log_reg_dense",),
    ("logitboost", None, None, (2020, "P", 0)),
    (
        "low_order_moms_dense",
        "low_order_moms_dense.csv",
        lambda r: np.vstack(
            (
                r.minimum,
                r.maximum,
                r.sum,
                r.sumSquares,
                r.sumSquaresCentered,
                r.mean,
                r.secondOrderRawMoment,
                r.variance,
                r.standardDeviation,
                r.variation,
            )
        ),
    ),
    (
        "low_order_moms_streaming",
        "low_order_moms_dense.csv",
        lambda r: np.vstack(
            (
                r.minimum,
                r.maximum,
                r.sum,
                r.sumSquares,
                r.sumSquaresCentered,
                r.mean,
                r.secondOrderRawMoment,
                r.variance,
                r.standardDeviation,
                r.variation,
            )
        ),
    ),
    ("multivariate_outlier", "multivariate_outlier.csv", lambda r: r[1].weights),
    ("naive_bayes", "naive_bayes.csv", lambda r: r[0].prediction),
    ("naive_bayes_streaming", "naive_bayes.csv", lambda r: r[0].prediction),
    ("normalization_minmax", "normalization_minmax.csv", "normalizedData"),
    ("normalization_zscore", "normalization_zscore.csv", "normalizedData"),
    ("pca", "pca.csv", "eigenvectors"),
    ("pca_transform", "pca_transform.csv", lambda r: r[1].transformedData),
    ("pivoted_qr", "pivoted_qr.csv", "matrixR"),
    ("quantiles", "quantiles.csv", "quantiles"),
    ("ridge_regression", "ridge_regression.csv", lambda r: r[0].prediction),
    ("ridge_regression_streaming", "ridge_regression.csv", lambda r: r[0].prediction),
    ("saga", None, None, (2019, "P", 3)),
    ("sgd_logistic_loss", "sgd_logistic_loss.csv", "minimum"),
    ("sgd_mse", "sgd_mse.csv", "minimum"),
    ("sorting",),
    ("stump_classification", None, None, (2020, "P", 0)),
    ("stump_regression", None, None, (2020, "P", 0)),
    ("svm_multiclass", "svm_multiclass.csv", lambda r: r[0].prediction),
    ("univariate_outlier", "univariate_outlier.csv", lambda r: r[1].weights),
    ("dbscan", "dbscan.csv", "assignments", (2019, "P", 5)),
    ("lasso_regression", None, None, (2019, "P", 5)),
    ("elastic_net", None, None, ((2020, "P", 1), (2021, "B", 105))),
]

for e in gen_examples:
    tmp = list(e)
    for i, elem in enumerate(e):
        if isinstance(elem, (tuple, NoneType)) or elem is None:
            tmp[i] = f"{elem}"
        else:
            tmp[i] = f'"{elem}"'

    print(f"TestArguments({', '.join(tmp)}),")
