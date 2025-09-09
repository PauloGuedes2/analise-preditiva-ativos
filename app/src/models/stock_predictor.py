from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class StockPredictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.best_params_clf = None
        self.best_params_reg = None
        self.features_used = None

    def train_models(self, X_train, X_train_scaled, y_class, y_price):
        """Treina ambos os modelos"""
        print("🎯 Treinando modelo de CLASSIFICAÇÃO...")

        param_grid_clf = {
            'n_estimators': [100, 150],
            'max_depth': [3, 4, 5],
            'min_samples_split': [10, 15],
            'class_weight': ['balanced']
        }

        grid_clf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid_clf,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy'
        )
        grid_clf.fit(X_train, y_class)
        self.classifier = grid_clf.best_estimator_
        self.best_params_clf = grid_clf.best_params_

        print("📈 Treinando modelo de REGRESSÃO...")

        param_grid_reg = {
            'n_estimators': [100, 150],
            'max_depth': [3, 4, 5],
            'min_samples_split': [10, 15]
        }

        grid_reg = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid_reg,
            cv=3,
            scoring='r2'
        )
        grid_reg.fit(X_train_scaled, y_price)
        self.regressor = grid_reg.best_estimator_
        self.best_params_reg = grid_reg.best_params_

        self.features_used = X_train.columns.tolist()
        print(f"✅ Features utilizadas no treino: {len(self.features_used)}")

    def evaluate_models(self, X_test, X_test_scaled, y_class, y_price):
        """Avalia ambos os modelos"""
        print("\n📊 AVALIAÇÃO DO MODELO DE CLASSIFICAÇÃO:")
        y_pred_class = self.classifier.predict(X_test)
        print(classification_report(y_class, y_pred_class))

        print("📈 AVALIAÇÃO DO MODELO DE REGRESSÃO:")
        y_pred_price = self.regressor.predict(X_test_scaled)

        mae = mean_absolute_error(y_price, y_pred_price)
        r2 = r2_score(y_price, y_pred_price)
        error_pct = (mae / y_price.mean()) * 100

        print(f"📊 Mean Absolute Error: R$ {mae:.2f}")
        print(f"📈 Error Percentage: {error_pct:.2f}%")
        print(f"🎯 R² Score: {r2:.4f}")

        print("\n🔍 EXEMPLOS DE PREVISÕES:")
        for i in range(min(5, len(y_price))):
            actual = y_price.iloc[i]
            predicted = y_pred_price[i]
            error = abs(actual - predicted)
            print(f"  Real: R$ {actual:.2f} | Previsto: R$ {predicted:.2f} | Erro: R$ {error:.2f}")

    def predict_next_day(self, X_last, X_last_scaled, current_price):
        """Faz previsão para o próximo dia"""
        direction_pred = self.classifier.predict(X_last)[0]
        direction_proba = self.classifier.predict_proba(X_last)[0]

        price_pred = self.regressor.predict(X_last_scaled)[0]

        if hasattr(current_price, 'iloc'):
            current_price_val = current_price.iloc[0] if hasattr(current_price, 'iloc') else float(current_price)
        else:
            current_price_val = float(current_price)

        expected_return = (price_pred / current_price_val) - 1

        return {
            'direction': 'ALTA' if direction_pred == 1 else 'QUEDA',
            'direction_confidence': direction_proba[1] if direction_pred == 1 else direction_proba[0],
            'predicted_price': price_pred,
            'current_price': current_price_val,
            'expected_return': expected_return,
            'price_change': price_pred - current_price_val
        }
