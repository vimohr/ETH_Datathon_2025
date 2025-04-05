import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib


class XGB1:
    def __init__(self, **xgboost_params):

        self.y_scaler = MinMaxScaler()

        self.pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("xgb", xgb.XGBRegressor(**xgboost_params)),
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

    def save(self, path):
        """
        Save the model to a file.
        Parameters:
        path : str
            Path to save the model.
        """
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.pipeline = joblib.load(path)
        print(f"Model loaded from {path}")
