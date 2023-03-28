from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

x_train, y_train, x_test, y_test = (
    np.load("./1090/x_train.npy"),
    np.load("./1090/y_train.npy"),
    np.load("./1090/x_test.npy"),
    np.load("./1090/y_test.npy"),
)

print(x_train.shape)
# out: (84545, 11)

from sklearn.ensemble import RandomForestRegressor as RFStock

from daal4py.sklearn.ensemble import RandomForestRegressor as RFOptimized

params = {
    "n_estimators": 150,
    "random_state": 44,
    "n_jobs": -1,
}

start = perf_counter()
rf = RFOptimized(**params).fit(x_train, y_train)
train_patched = perf_counter() - start
print(f"Intel® extension for Scikit-learn time: {train_patched:.2f} s")
y_pred_opt = rf.predict(x_test)
mse_opt = metrics.mean_squared_error(y_test, y_pred_opt)
print(f"Intel® extension for Scikit-learn Mean Squared Error: {mse_opt}")

start = perf_counter()
rf = RFStock(**params).fit(x_train, y_train)
train_patched = perf_counter() - start
print(f"stock-learn time: {train_patched:.2f} s")
y_pred = rf.predict(x_test)
mse_opt = metrics.mean_squared_error(y_test, y_pred)
print(f"stock-learn Mean Squared Error: {mse_opt}")

fig, ax = plt.subplots()

ax.scatter(y_test, y_test, label="ground truth", s=2, c="black")
ax.scatter(y_test, y_pred_opt, label="sklex", s=2, c="g", alpha=0.2)
ax.scatter(y_test, y_pred, label="stock", s=2, c="r", alpha=0.2)

ax.legend()

plt.savefig("1090/y_test.png")
