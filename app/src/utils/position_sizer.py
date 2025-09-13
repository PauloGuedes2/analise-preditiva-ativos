class PositionSizer:
    def __init__(self, max_risk_per_trade=0.02, max_portfolio_risk=0.10):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk

    def calculate_position_size(self, current_price, stop_loss, portfolio_value, probability):
        """
        Calcula tamanho da posição baseado em:
        - Preço atual
        - Stop loss
        - Valor do portfólio
        - Probabilidade da previsão
        """
        # Risco por unidade
        risk_per_share = current_price - stop_loss

        # Ajustar risco baseado na probabilidade
        confidence_factor = min(probability / 0.6, 1.0)  # Normalizado para prob > 0.6

        # Risco máximo ajustado
        max_risk_amount = portfolio_value * self.max_risk_per_trade * confidence_factor

        # Calcular número de shares
        if risk_per_share > 0:
            position_size = max_risk_amount / risk_per_share
            return int(position_size)
        return 0

    def calculate_stop_loss(self, current_price, volatility, trend_direction=1):
        """
        Calcula stop loss dinâmico
        trend_direction: 1 para long, -1 para short
        """
        # Stop baseado em ATR (Average True Range)
        atr_based_stop = current_price - (2 * volatility * trend_direction)

        # Stop percentual
        percent_stop = current_price * (1 - 0.02 * trend_direction)  # 2%

        # Usar o mais conservador
        if trend_direction == 1:
            return max(atr_based_stop, percent_stop)
        else:
            return min(atr_based_stop, percent_stop)