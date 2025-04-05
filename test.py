from src.models.decisiontrees.xgboost import XGB1
from src.models.decisiontrees.catboost import CatBoost1
from src.models.decisiontrees.lightgbm import LGBM1
from utils import get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import webbrowser


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
)

X_train = X[0][:-800]
y_train = y[0][:-800]
X_test = X[0][-800:]
y_test = y[0][-800:]
times_train = times[0][:-800]
times_test = times[0][-800:]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(times_train.shape)
print(times_test.shape)

model = XGB1()
model2 = CatBoost1(loss_function="MultiRMSE")
model3 = LGBM1()

model2.fit(X_train, y_train)
