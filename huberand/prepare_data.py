from sklearn.datasets import fetch_openml

data, target = fetch_openml(data_id=42759, return_X_y=True, data_home="../scikit_learn_data")
categorical_features = list(set(data.columns) - set(data._get_numeric_data().columns))
data = data.drop(columns=categorical_features)
data.to_pickle("data.pkl")
target.to_pickle("target.pkl")
