from src.models.decisiontrees.xgboost import XGB1
from src.models.decisiontrees.catboost import CatBoost1
from src.models.decisiontrees.lightgbm import LGBM1
from utils import get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

len_week = 24 * 7
X, y, times = get_data(
    "ES",
    indices=[114],
    demand=[0]
    + [i for i in range(1, 8000, 6)]
    + [i for i in range(6300, 6400)]
    + [i for i in range(8700, 8800)],
    temp=[0, 1, 24, 24 * 7, 24 * 365],
    spv=[0, 1, 24, 24 * 7, 24 * 365],
    rollout_values=[-i for i in range(25)],
    n_futures=24 * 31,
)

X_train = X[0][:-800]
y_train = y[0][:-800]
X_test = X[0][-800:]
y_test = y[0][-800:]
times_train = times[0][:-800]
times_test = times[0][-800:]

model = XGB1()
# model2 = CatBoost1(loss_function="MultiRMSE", task_type="GPU", devices="0:1:2:3")
model3 = LGBM1()

model3.fit(X_train, y_train)

fig = plt.figure(figsize=(200, 10))
plt.plot(times_test, y_test[:, -1], label="Real")
plt.plot(times_test, model3.predict(X_test)[:, -1], label="CatBoost")
fig = plt.savefig("catboost1.png")

plt.figure(figsize=(200, 10))
plt.plot(times_test, y_test[:, 1], label="Real")
plt.plot(times_test, model3.predict(X_test)[:, 1], label="CatBoost")
fig = plt.savefig("catboost2.png")

plt.figure(figsize=(200, 10))
plt.plot(times_test, y_test[:, 2], label="Real")
plt.plot(times_test, model3.predict(X_test)[:, 2], label="CatBoost")
fig = plt.savefig("catboost3.png")

plt.figure(figsize=(200, 10))
plt.plot(times_test, y_test[:, 3], label="Real")
plt.plot(times_test, model3.predict(X_test)[:, 3], label="CatBoost")
fig = plt.savefig("catboost4.png")

plt.figure(figsize=(200, 10))
plt.plot(times_train, y_train[:, 0], label="Real")
plt.plot(times_test, y_test[:, 0], label="CatBoost")
fig = plt.savefig("catboost5.png")
