from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib


class CatBoost1:

    def __init__(self, **CatParam):
        self.pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("catb", CatBoostRegressor(**CatParam)),
            ]
        )
        self.y_scaler = MinMaxScaler()

    def fit(self, X, y):
        self.y_scaler.fit_transform(y)

        self.pipeline.fit(X, y)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def rescale(self, y_pred):
        """
        Rescale the data using the fitted MinMaxScaler.
        Parameters:
        X : pd.DataFrame(single row in table)
        """
        return self.y_scaler.inverse_transform(y_pred)

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
