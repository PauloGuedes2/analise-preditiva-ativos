import warnings
from datetime import datetime, timedelta

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import DataLoader
from src.models.feature_engineer import FeatureEngineer
from src.models.stock_predictor import StockPredictor
from src.utils.risk_analyzer import RiskAnalyzer

warnings.filterwarnings('ignore')


class StockPredictionSystem:
    def __init__(self):
        self.predictor = StockPredictor()
        self.feature_engineer = FeatureEngineer()
        self.data_loader = DataLoader()
        self.risk_analyzer = RiskAnalyzer()
        self.scaler = StandardScaler()

    def prepare_for_prediction(self, recent_data, features_used):
        """Prepara dados recentes para predição usando features do treino"""
        recent_with_features = self.feature_engineer.create_features(recent_data)
        X_recent, _, _, _, _ = self.feature_engineer.prepare_training_data(recent_with_features)

        if features_used:
            available_features = [str(f[0]) if isinstance(f, tuple) else str(f) for f in features_used]
            X_recent = X_recent[available_features]

        X_recent_scaled = self.scaler.transform(X_recent)
        X_recent_scaled = pd.DataFrame(X_recent_scaled, columns=X_recent.columns, index=X_recent.index)

        return X_recent, X_recent_scaled

    def print_detailed_analysis(self, y_test, y_pred, returns_test, prediction):
        """Imprime análise detalhada"""
        print("\n" + "=" * 70)
        print("📊 ANÁLISE DETALHADA DE PERFORMANCE")
        print("=" * 70)

        risk_metrics = self.risk_analyzer.calculate_risk_metrics(y_test, y_pred, returns_test)

        print(f"🎯 Taxa de Acerto: {risk_metrics['accuracy']:.2%}")
        print(f"💰 Lucro Acumulado: {risk_metrics['total_profit']:.2%}")
        print(f"📈 Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
        print(f"✅ Win Rate: {risk_metrics['win_rate']:.2%}")
        print(f"🎯 Profit Factor: {risk_metrics['profit_factor']:.2f}")

        print(f"\n📊 Estatísticas dos Retornos:")
        print(f"   Média: {returns_test.mean():.4%}")
        print(f"   Volatilidade: {returns_test.std():.4%}")
        print(f"   Melhor Dia: {returns_test.max():.2%}")
        print(f"   Pior Dia: {returns_test.min():.2%}")

        print("\n" + "=" * 70)
        print("🎰 SINAIS DE TRADING RECOMENDADOS")
        print("=" * 70)

        signals = self.risk_analyzer.generate_trading_signals(prediction)
        for signal in signals:
            print(f"   {signal}")

        print(f"\n📊 Previsão de Preço com Intervalo:")
        current_price = prediction['current_price']
        predicted_price = prediction['predicted_price']
        error_margin = abs(predicted_price - current_price) * 0.3

        print(f"   💰 Preço Atual: R$ {current_price:.2f}")
        print(f"   📈 Preço Previsto: R$ {predicted_price:.2f}")
        print(
            f"   📍 Intervalo Provável: R$ {predicted_price - error_margin:.2f} - R$ {predicted_price + error_margin:.2f}")
        print(f"   🎯 Alvo de Ganho: {prediction['expected_return']:.2%}")

        print(f"\n⚠️  ALERTAS DE MERCADO:")
        volatility = returns_test.std()
        if volatility > 0.03:
            print("   🔥 ALTA VOLATILIDADE - Mercado instável")
        if risk_metrics['max_drawdown'] < -0.1:
            print("   🚨 DRAWDOWN ELEVADO - Cuidado com risco")
        if risk_metrics['sharpe_ratio'] < 0:
            print("   ⚠️  SHARPE NEGATIVO - Estratégia não lucrativa")

    def run(self):
        """Executa o sistema completo de previsão"""
        TICKER = "AMZO34.SA"
        END_DATE = datetime.now().strftime('%Y-%m-%d')
        START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        print("🚀 INICIANDO SISTEMA COMPLETO DE PREVISÃO")
        print(f"📊 Ativo: {TICKER}")
        print(f"📅 Período: {START_DATE} a {END_DATE}")
        print("=" * 60)

        try:
            data = self.data_loader.get_data_with_fallback(TICKER, START_DATE, END_DATE)

            print(f"✅ Dados carregados: {len(data)} registros")
            print(f"📅 Período: {data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}")

            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()

            print(f"📊 Split temporal - Treino: {len(train_data)}, Teste: {len(test_data)}")
            print(
                f"📅 Treino: {train_data.index.min().strftime('%Y-%m-%d')} a {train_data.index.max().strftime('%Y-%m-%d')}")
            print(
                f"📈 Teste: {test_data.index.min().strftime('%Y-%m-%d')} a {test_data.index.max().strftime('%Y-%m-%d')}")

            print("🛠️  Criando features para dados de TREINO...")
            train_with_features = self.feature_engineer.create_features(train_data)

            print("🛠️  Criando features para dados de TESTE...")
            test_with_features = self.feature_engineer.create_features(test_data)

            X_train, y_class_train, y_price_train, _, features = self.feature_engineer.prepare_training_data(
                train_with_features)
            X_test, y_class_test, y_price_test, y_return_test, _ = self.feature_engineer.prepare_training_data(
                test_with_features)

            X_test = X_test[features]
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

            print(f"📋 Features utilizadas: {len(features)}")
            print(f"📊 Treino: {len(X_train)} amostras")
            print(f"📈 Teste: {len(X_test)} amostras")

            self.predictor.train_models(X_train, X_train_scaled, y_class_train, y_price_train)
            self.predictor.evaluate_models(X_test, X_test_scaled, y_class_test, y_price_test)

            current_price = data['Close'].iloc[-1]
            last_day_data = data.iloc[-30:].copy()
            X_last, X_last_scaled = self.prepare_for_prediction(last_day_data, features)

            X_last = X_last.iloc[[-1]]
            X_last_scaled = X_last_scaled.iloc[[-1]]

            prediction = self.predictor.predict_next_day(X_last, X_last_scaled, current_price)

            print("\n" + "=" * 60)
            print("🔮 PREVISÃO PARA O PRÓXIMO DIA")
            print("=" * 60)
            print(f"📈 Direção: {prediction['direction']}")
            print(f"🎯 Confiança na direção: {prediction['direction_confidence']:.2%}")
            print(f"💰 Preço atual: R$ {prediction['current_price']:.2f}")
            print(f"📊 Preço previsto: R$ {prediction['predicted_price']:.2f}")
            print(f"🔼 Variação esperada: R$ {prediction['price_change']:.2f}")
            print(f"📈 Retorno esperado: {prediction['expected_return']:.2%}")

            print("\n📈 CALCULANDO MÉTRICAS AVANÇADAS...")
            y_pred_class = self.predictor.classifier.predict(X_test)
            self.print_detailed_analysis(y_class_test, y_pred_class, y_return_test, prediction)

            print("\n" + "=" * 70)
            print("🔮 PREVISÃO PARA OS PRÓXIMOS 3 DIAS")
            print("=" * 70)

            for days_ahead in [1, 2, 3]:
                try:
                    recent_data = data.iloc[-(30 + days_ahead):-days_ahead].copy() if days_ahead > 0 else data.iloc[
                                                                                                          -30:].copy()
                    X_future, X_future_scaled = self.prepare_for_prediction(recent_data, features)

                    X_day = X_future.iloc[[-1]]
                    X_day_scaled = X_future_scaled.iloc[[-1]]
                    current_price_day = data['Close'].iloc[-(days_ahead + 1)] if days_ahead > 0 else data['Close'].iloc[
                        -1]

                    future_pred = self.predictor.predict_next_day(X_day, X_day_scaled, current_price_day)

                    print(f"📅 Dia +{days_ahead}: {future_pred['direction']} "
                          f"(Retorno: {future_pred['expected_return']:.2%}, "
                          f"Conf: {future_pred['direction_confidence']:.2%})")

                except Exception as e:
                    print(f"📅 Dia +{days_ahead}: Erro na previsão - {e}")

            print("\n⚠️  ANÁLISE DE RISCO:")
            current_volatility = y_return_test.std()
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
    system = StockPredictionSystem()
    system.run()
