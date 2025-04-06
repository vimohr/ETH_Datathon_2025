from carlo_utils import get_data
from src.models.decisiontrees.lightgbm import LGBM1

import pandas as pd
import numpy as np

all_data = get_data(
    demand=[0]
    + [i for i in range(1, 8000, 6)]
    + [i for i in range(6300, 6400)]
    + [i for i in range(8700, 8800)],
    temp=[0, 1, 2, 3, 4, 5, 6, 24, 24 * 7, 24 * 365],
    spv=[0, 2, 3, 4, 5, 6, 1, 24, 24 * 7, 24 * 365],
    n_futures=31 * 24,
    rollout_values=[i for i in range(30)],
)


cutoff = pd.to_datetime("2024-06-01")

train_data = all_data[all_data.index < cutoff]
test_data = all_data[all_data.index >= cutoff]

forecast_columns = train_data.columns[train_data.columns.str.contains("forecast")]
X_train = train_data.drop(columns=forecast_columns)
y_train = train_data[forecast_columns]

X_test = test_data.drop(columns=forecast_columns)
y_test = test_data[forecast_columns]

X_train.shape, y_train.shape, X_test.shape, y_test.shape

from src.models.decisiontrees.lightgbm import LGBM1

model = LGBM1()
