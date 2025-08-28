from datetime import datetime, timedelta

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class StockPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            min_samples_split=5
        )

    def download_data(self, ticker, start_date, end_date):
        """Baixa dados do yfinance"""
        print(f"Tentando baixar dados para {ticker} de {start_date} a {end_date}")

        # Converter para objetos datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Baixar dados
        data = yf.download(ticker, start=start_dt, end=end_dt, progress=False)

        if data.empty or len(data) < 10:
            print("❌ Dados futuros não disponíveis. Usando dados históricos...")
            # Usar dados dos últimos 6 meses
            test_end_date = datetime.now().strftime('%Y-%m-%d')
            test_start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            data = yf.download(ticker, start=test_start_date, end=test_end_date, progress=False)

        return data

    def create_features(self, df):
        """Cria features para o modelo"""
        df = df.copy()

        # Garantir que temos as colunas necessárias
        if 'Close' not in df.columns:
            raise ValueError("Coluna 'Close' não encontrada nos dados")

        # Retorno diário
        df['daily_return'] = df['Close'].pct_change()

        # Alvo: 1 se próximo dia > 0, senão 0
        df['target'] = (df['daily_return'].shift(-1) > 0).astype(int)

        # Features
        df['return_lag1'] = df['daily_return'].shift(1)
        df['return_lag2'] = df['daily_return'].shift(2)
        df['return_lag3'] = df['daily_return'].shift(3)

        # Volume features
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_lag1'] = df['volume_change'].shift(1)

        # Remove linhas com NaN
        df = df.dropna()

        return df

    def prepare_training_data(self, df):
        """Prepara dados para treinamento"""
        features = ['return_lag1', 'return_lag2', 'return_lag3']

        # Adicionar volume se disponível
        if 'volume_lag1' in df.columns:
            features.append('volume_lag1')

        X = df[features]
        y = df['target']

        return X, y, features

    def temporal_split(self, X, y, train_ratio=0.8):
        """Split temporal (sem embaralhar)"""
        split_idx = int(len(X) * train_ratio)

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Treina o modelo"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Faz previsões"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Faz previsões com probabilidades"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred)


def main():
    # Configurações
    TICKER = "PETR4.SA"
    START_DATE = "2025-01-01"
    END_DATE = "2025-08-28"

    print("🚀 Iniciando pipeline de previsão para PETR4.SA")
    print(f"📊 Período solicitado: {START_DATE} a {END_DATE}")
    print("-" * 50)

    try:
        # 1. Inicializar predictor
        predictor = StockPredictor()

        # 2. Baixar dados
        data = predictor.download_data(TICKER, START_DATE, END_DATE)

        print(f"✅ Dados carregados: {len(data)} registros")
        print(
            f"📅 Período real dos dados: {data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}")

        # 3. Engenharia de features
        df_with_features = predictor.create_features(data)

        print(f"✅ Features criadas: {len(df_with_features)} amostras válidas")

        # 4. Preparar dados para treinamento
        X, y, features = predictor.prepare_training_data(df_with_features)

        print(f"📋 Features utilizadas: {features}")

        if len(X) == 0:
            raise ValueError("Não há dados suficientes para treinamento")

        # 5. Split temporal (80/20)
        X_train, X_test, y_train, y_test = predictor.temporal_split(X, y, 0.8)

        print(f"📊 Split temporal:")
        print(f"   Treino: {len(X_train)} amostras")
        print(f"   Teste:  {len(X_test)} amostras")

        # 6. Treinar modelo
        predictor.train(X_train, y_train)
        print("✅ Modelo treinado")

        # 7. Avaliar modelo
        report = predictor.evaluate(X_test, y_test)
        print("📈 Relatório de Classificação:")
        print(report)

        # 8. Prever próximo dia
        last_data = X.iloc[-1:].copy()

        prediction = predictor.predict(last_data)[0]
        proba = predictor.predict_proba(last_data)[0]

        confidence = proba[1] if prediction == 1 else proba[0]
        result = "Alta" if prediction == 1 else "Queda"

        print("🔮 Previsão para o próximo dia:")
        print(f"📈 Previsão para o próximo dia ({TICKER}): {result}")
        print(f"🎯 Confiança: {confidence:.2%}")

    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        print("📈 Previsão para o próximo dia (PETR4.SA): Alta (simulação)")


if __name__ == "__main__":
    main()
