import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class XGB1:
    def __init__(self, **xgboost_params):
        self.xgboost_params = xgboost_params

        self.pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("xgb", xgb.XGBRegressor(**self.xgboost_params)),
            ]
        )

    def fit(self, X, y):
        """
        Fit the model to the training data.
        Parameters:
        X : pd.DataFrame(single row in table)
        y : pd.Series(target variable)
        """

        self.pipeline.fit(X, y)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)
