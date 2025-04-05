from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib


class LGBM1:

    def __init__(self, **CatParam):
        self.pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("lgbm", MultiOutputRegressor(LGBMRegressor(**CatParam))),
            ]
        )

    def fit(self, X, y):
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
