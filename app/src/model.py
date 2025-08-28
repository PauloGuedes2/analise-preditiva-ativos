from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np


class StockPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            min_samples_split=5
        )
        self.features = None

    def train(self, x_train, y_train):
        """Treina o modelo"""
        self.model.fit(x_train, y_train)
        self.features = x_train.columns.tolist()

    def predict(self, x):
        """Faz previsões"""
        return self.model.predict(x)

    def predict_proba(self, x):
        """Faz previsões com probabilidades"""
        return self.model.predict_proba(x)

    def evaluate(self, x_test, y_test):
        """Avalia o modelo"""
        y_pred = self.predict(x_test)
        return classification_report(y_test, y_pred)

    def temporal_split(self, x, y, train_ratio=0.8):
        """Split temporal (sem embaralhar)"""
        split_idx = int(len(x) * train_ratio)

        x_train = x.iloc[:split_idx]
        x_test = x.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        return x_train, x_test, y_train, y_test