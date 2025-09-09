import numpy as np
from sklearn.metrics import accuracy_score


class RiskAnalyzer:
    @staticmethod
    def calculate_risk_metrics(y_test, y_pred, returns_test):
        """Calcula métricas de risco e performance"""
        results = {}

        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['total_profit'] = returns_test[y_pred == 1].sum()

        excess_returns = returns_test - 0.0001
        results['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        cumulative_returns = (1 + returns_test).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        results['max_drawdown'] = drawdown.min()

        winning_trades = returns_test[y_pred == 1] > 0
        results['win_rate'] = winning_trades.mean() if len(winning_trades) > 0 else 0

        gross_profit = returns_test[(y_pred == 1) & (returns_test > 0)].sum()
        gross_loss = abs(returns_test[(y_pred == 1) & (returns_test < 0)].sum())
        results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return results

    @staticmethod
    def generate_trading_signals(prediction):
        """Gera sinais de trading baseado na previsão"""
        signals = []

        if prediction['direction'] == 'ALTA':
            signals.append("📈 SINAL: COMPRA")
        else:
            signals.append("📉 SINAL: VENDA")

        confidence = prediction['direction_confidence']
        if confidence > 0.7:
            signals.append("💪 FORTE (Confiança > 70%)")
        elif confidence > 0.6:
            signals.append("👍 MÉDIO (Confiança 60-70%)")
        else:
            signals.append("⚠️  FRACO (Confiança < 60%)")

        expected_return = prediction['expected_return']
        if expected_return > 0.015:
            signals.append("🎯 ALTO POTENCIAL (Retorno > 1.5%)")
        elif expected_return > 0.005:
            signals.append("✅ OPERAR (Retorno 0.5-1.5%)")
        elif expected_return > -0.005:
            signals.append("⏸️  NEUTRO (Retorno -0.5% a 0.5%)")
        else:
            signals.append("🚫 EVITAR (Retorno < -0.5%)")

        if confidence > 0.65 and abs(expected_return) > 0.008:
            position_size = "Tamanho: NORMAL"
        elif confidence > 0.75 and abs(expected_return) > 0.015:
            position_size = "Tamanho: MAIOR"
        else:
            position_size = "Tamanho: REDUZIDO"
        signals.append(position_size)

        return signals
