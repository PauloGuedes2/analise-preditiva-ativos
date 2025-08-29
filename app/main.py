import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class StockPredictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.best_params_clf = None
        self.best_params_reg = None

    def download_data(self, ticker, start_date, end_date):
        """Baixa dados do yfinance"""
        print(f"Tentando baixar dados para {ticker}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        data = yf.download(ticker, start=start_dt, end=end_dt, progress=False)

        if data.empty or len(data) < 10:
            print("❌ Dados não disponíveis. Usando dados históricos...")
            test_end_date = datetime.now().strftime('%Y-%m-%d')
            test_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            data = yf.download(ticker, start=test_start_date, end=test_end_date, progress=False)

        return data

    def create_features(self, df):
        """Cria features para ambos os modelos"""
        df = df.copy()

        # Features básicas
        df['daily_return'] = df['Close'].pct_change()

        # Médias Móveis
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

        df['MA_ratio_5_20'] = df['MA_5'] / df['MA_20']
        df['MA_ratio_10_50'] = df['MA_10'] / df['MA_50']

        # Volatilidade
        df['volatility_10'] = df['daily_return'].rolling(window=10).std()
        df['volatility_20'] = df['daily_return'].rolling(window=20).std()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Volume features
        if 'Volume' in df.columns:
            df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']
            df['volume_change'] = df['Volume'].pct_change()

        # Retornos defasados
        for lag in [1, 2, 3, 5, 7]:
            df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

        # ALVO PARA CLASSIFICAÇÃO (Alta/Queda)
        df['target_class'] = (df['daily_return'].shift(-1) > 0).astype(int)

        # ALVO PARA REGRESSÃO (Preço Futuro)
        df['target_price'] = df['Close'].shift(-1)  # Preço do próximo dia
        df['target_return'] = df['daily_return'].shift(-1)  # Retorno do próximo dia

        # Remove linhas com NaN
        df = df.dropna()

        return df

    def prepare_training_data(self, df):
        """Prepara dados para treinamento"""
        # Features selecionadas
        features = [
            'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_ratio_5_20', 'MA_ratio_10_50',
            'volatility_10', 'volatility_20', 'RSI', 'MACD', 'MACD_signal',
            'volume_ratio', 'volume_change', 'return_lag_1', 'return_lag_2', 'return_lag_3'
        ]

        # Manter apenas features disponíveis
        available_features = [f for f in features if f in df.columns]

        X = df[available_features]
        y_class = df['target_class']  # Para classificação
        y_price = df['target_price']  # Para regressão (preço)
        y_return = df['target_return']  # Para regressão (retorno)

        # Normalizar features para regressão
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=available_features, index=X.index)

        return X, X_scaled, y_class, y_price, y_return, available_features

    def temporal_split(self, X, y, train_ratio=0.8):
        """Split temporal"""
        split_idx = int(len(X) * train_ratio)

        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        else:
            X_train, X_test = X[:split_idx], X[split_idx:]

        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, X_train_scaled, y_class, y_price):
        """Treina ambos os modelos"""
        print("🎯 Treinando modelo de CLASSIFICAÇÃO...")

        # Otimizar classificador
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

        # Otimizar regressor
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

        # Mostrar exemplos de previsões
        print("\n🔍 EXEMPLOS DE PREVISÕES:")
        for i in range(min(5, len(y_price))):
            actual = y_price.iloc[i]
            predicted = y_pred_price[i]
            error = abs(actual - predicted)
            print(f"  Real: R$ {actual:.2f} | Previsto: R$ {predicted:.2f} | Erro: R$ {error:.2f}")

    def predict_next_day(self, X_last, X_last_scaled, current_price):
        """Faz previsão para o próximo dia"""
        # Previsão de classificação
        direction_pred = self.classifier.predict(X_last)[0]
        direction_proba = self.classifier.predict_proba(X_last)[0]

        # Previsão de preço
        price_pred = self.regressor.predict(X_last_scaled)[0]

        # 🔥 CORREÇÃO AQUI: Garantir que current_price é um número float
        if hasattr(current_price, 'iloc'):
            current_price_val = current_price.iloc[0] if hasattr(current_price, 'iloc') else float(current_price)
        else:
            current_price_val = float(current_price)

        # Calcular retorno esperado
        expected_return = (price_pred / current_price_val) - 1

        return {
            'direction': 'ALTA' if direction_pred == 1 else 'QUEDA',
            'direction_confidence': direction_proba[1] if direction_pred == 1 else direction_proba[0],
            'predicted_price': price_pred,
            'current_price': current_price_val,
            'expected_return': expected_return,
            'price_change': price_pred - current_price_val
        }

    def calculate_risk_metrics(self, y_test, y_pred, returns_test):
        """Calcula métricas de risco e performance"""
        results = {}

        # Taxa de acerto
        results['accuracy'] = accuracy_score(y_test, y_pred)

        # Lucro/Prejuízo acumulado
        results['total_profit'] = returns_test[y_pred == 1].sum()

        # Sharpe Ratio
        excess_returns = returns_test - 0.0001  # Taxa livre de risco diária
        results['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Maximum Drawdown
        cumulative_returns = (1 + returns_test).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        results['max_drawdown'] = drawdown.min()

        # Win Rate
        winning_trades = returns_test[y_pred == 1] > 0
        results['win_rate'] = winning_trades.mean() if len(winning_trades) > 0 else 0

        # Profit Factor
        gross_profit = returns_test[(y_pred == 1) & (returns_test > 0)].sum()
        gross_loss = abs(returns_test[(y_pred == 1) & (returns_test < 0)].sum())
        results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return results

    def generate_trading_signals(self, prediction):
        """Gera sinais de trading baseado na previsão"""
        signals = []

        # Sinal de Direção
        if prediction['direction'] == 'ALTA':
            signals.append("📈 SINAL: COMPRA")
        else:
            signals.append("📉 SINAL: VENDA")

        # Força do Sinal
        confidence = prediction['direction_confidence']
        if confidence > 0.7:
            signals.append("💪 FORTE (Confiança > 70%)")
        elif confidence > 0.6:
            signals.append("👍 MÉDIO (Confiança 60-70%)")
        else:
            signals.append("⚠️  FRACO (Confiança < 60%)")

        # Recomendação de Operação
        expected_return = prediction['expected_return']
        if expected_return > 0.015:
            signals.append("🎯 ALTO POTENCIAL (Retorno > 1.5%)")
        elif expected_return > 0.005:
            signals.append("✅ OPERAR (Retorno 0.5-1.5%)")
        elif expected_return > -0.005:
            signals.append("⏸️  NEUTRO (Retorno -0.5% a 0.5%)")
        else:
            signals.append("🚫 EVITAR (Retorno < -0.5%)")

        # tamanho da posição sugerida
        if confidence > 0.65 and abs(expected_return) > 0.008:
            position_size = "Tamanho: NORMAL"
        elif confidence > 0.75 and abs(expected_return) > 0.015:
            position_size = "Tamanho: MAIOR"
        else:
            position_size = "Tamanho: REDUZIDO"
        signals.append(position_size)

        return signals

    def print_detailed_analysis(self, y_test, y_pred, returns_test, prediction):
        """Imprime análise detalhada"""

        print("\n" + "=" * 70)
        print("📊 ANÁLISE DETALHADA DE PERFORMANCE")
        print("=" * 70)

        # Métricas de Risco
        risk_metrics = self.calculate_risk_metrics(y_test, y_pred, returns_test)

        print(f"🎯 Taxa de Acerto: {risk_metrics['accuracy']:.2%}")
        print(f"💰 Lucro Acumulado: {risk_metrics['total_profit']:.2%}")
        print(f"📈 Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
        print(f"✅ Win Rate: {risk_metrics['win_rate']:.2%}")
        print(f"🎯 Profit Factor: {risk_metrics['profit_factor']:.2f}")

        # Análise Estatística
        print(f"\n📊 Estatísticas dos Retornos:")
        print(f"   Média: {returns_test.mean():.4%}")
        print(f"   Volatilidade: {returns_test.std():.4%}")
        print(f"   Melhor Dia: {returns_test.max():.2%}")
        print(f"   Pior Dia: {returns_test.min():.2%}")

        # Sinais de Trading
        print("\n" + "=" * 70)
        print("🎰 SINAIS DE TRADING RECOMENDADOS")
        print("=" * 70)

        signals = self.generate_trading_signals(prediction)
        for signal in signals:
            print(f"   {signal}")

        # Previsão de Preço com Intervalo de Confiança
        print(f"\n📊 Previsão de Preço com Intervalo:")
        current_price = prediction['current_price']
        predicted_price = prediction['predicted_price']
        error_margin = abs(predicted_price - current_price) * 0.3  # 30% do erro como margem

        print(f"   💰 Preço Atual: R$ {current_price:.2f}")
        print(f"   📈 Preço Previsto: R$ {predicted_price:.2f}")
        print(
            f"   📍 Intervalo Provável: R$ {predicted_price - error_margin:.2f} - R$ {predicted_price + error_margin:.2f}")
        print(f"   🎯 Alvo de Ganho: {prediction['expected_return']:.2%}")

        # Alertas de Mercado
        print(f"\n⚠️  ALERTAS DE MERCADO:")
        volatility = returns_test.std()
        if volatility > 0.03:
            print("   🔥 ALTA VOLATILIDADE - Mercado instável")
        if risk_metrics['max_drawdown'] < -0.1:
            print("   🚨 DRAWDOWN ELEVADO - Cuidado com risco")
        if risk_metrics['sharpe_ratio'] < 0:
            print("   ⚠️  SHARPE NEGATIVO - Estratégia não lucrativa")


def main():
    # Configurações
    TICKER = "PETR4.SA"
    END_DATE = datetime.now().strftime('%Y-%m-%d')  # Hoje
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 anos atrás

    print("🚀 INICIANDO SISTEMA COMPLETO DE PREVISÃO")
    print(f"📊 Ativo: {TICKER}")
    print(f"📅 Período: {START_DATE} a {END_DATE}")
    print("=" * 60)

    try:
        # 1. Inicializar predictor
        predictor = StockPredictor()

        # 2. Baixar dados
        data = predictor.download_data(TICKER, START_DATE, END_DATE)

        print(f"✅ Dados carregados: {len(data)} registros")
        print(f"📅 Período: {data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}")

        # 3. Engenharia de features
        df_with_features = predictor.create_features(data)
        print(f"✅ Features criadas: {len(df_with_features)} amostras")

        # 4. Preparar dados
        X, X_scaled, y_class, y_price, y_return, features = predictor.prepare_training_data(df_with_features)
        print(f"📋 Features utilizadas: {len(features)}")

        # 5. Split temporal
        X_train, X_test, y_class_train, y_class_test = predictor.temporal_split(X, y_class)
        X_train_scaled, X_test_scaled, y_price_train, y_price_test = predictor.temporal_split(X_scaled, y_price)

        print(f"📊 Split temporal - Treino: {len(X_train)}, Teste: {len(X_test)}")

        train_dates = f"{X_train.index.min().strftime('%Y-%m-%d')} a {X_train.index.max().strftime('%Y-%m-%d')}"
        test_dates = f"{X_test.index.min().strftime('%Y-%m-%d')} a {X_test.index.max().strftime('%Y-%m-%d')}"

        print(f"📅 Período de Treino: {train_dates}")
        print(f"📈 Período de Teste: {test_dates}")

        # 6. Treinar modelos
        predictor.train_models(X_train, X_train_scaled, y_class_train, y_price_train)

        # 7. Avaliar modelos
        predictor.evaluate_models(X_test, X_test_scaled, y_class_test, y_price_test)

        # 8. Prever próximo dia
        current_price = data['Close'].iloc[-1]
        last_features = X.iloc[-1:].copy()
        last_features_scaled = X_scaled.iloc[-1:].copy()

        prediction = predictor.predict_next_day(last_features, last_features_scaled, current_price)

        print("\n" + "=" * 60)
        print("🔮 PREVISÃO PARA O PRÓXIMO DIA")
        print("=" * 60)
        print(f"📈 Direção: {prediction['direction']}")
        print(f"🎯 Confiança na direção: {prediction['direction_confidence']:.2%}")
        print(f"💰 Preço atual: R$ {prediction['current_price']:.2f}")
        print(f"📊 Preço previsto: R$ {prediction['predicted_price']:.2f}")
        print(f"🔼 Variação esperada: R$ {prediction['price_change']:.2f}")
        print(f"📈 Retorno esperado: {prediction['expected_return']:.2%}")

        # 9. Análise detalhada e métricas
        print("\n📈 CALCULANDO MÉTRICAS AVANÇADAS...")

        # Calcular retornos reais do período de teste
        returns_test = df_with_features['target_return'].iloc[-len(y_class_test):]

        # Fazer previsões completas para o teste
        y_pred_class = predictor.classifier.predict(X_test)

        # Gerar análise detalhada
        predictor.print_detailed_analysis(y_class_test, y_pred_class, returns_test, prediction)

        # 10. Previsão para os próximos dias
        print("\n" + "=" * 70)
        print("🔮 PREVISÃO PARA OS PRÓXIMOS 3 DIAS")
        print("=" * 70)

        # Simular previsões para os próximos dias (usando dados recentes)
        for days_ahead in [1, 2, 3]:
            try:
                future_features = X.iloc[-days_ahead:].copy()
                future_features_scaled = X_scaled.iloc[-days_ahead:].copy()
                current_price = data['Close'].iloc[-days_ahead]

                future_pred = predictor.predict_next_day(
                    future_features.iloc[[0]],
                    future_features_scaled.iloc[[0]],
                    current_price
                )

                print(f"📅 Dia +{days_ahead}: {future_pred['direction']} "
                      f"(Retorno: {future_pred['expected_return']:.2%}, "
                      f"Conf: {future_pred['direction_confidence']:.2%})")

            except:
                print(f"📅 Dia +{days_ahead}: Dados insuficientes")

        # Análise de risco
        print("\n⚠️  ANÁLISE DE RISCO:")
        current_volatility = returns_test.std()
        if current_volatility > 0.02:
            print(f"⚠️  Volatilidade atual: {current_volatility:.2%} - Mercado agitado")
        if prediction['expected_return'] > 0.01:
            print("✅ Retorno esperado positivo - Potencial de ganho")
        else:
            print("⚠️  Retorno esperado baixo - Considerar não operar")

        if abs(prediction['expected_return']) > 0.03:
            print("⚠️  Alta volatilidade esperada - Cuidado com risco")

    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
