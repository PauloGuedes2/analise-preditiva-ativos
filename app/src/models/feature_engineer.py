"""
Feature engineering e utilitários refinados.
Mantém o pipeline de criação de features.
"""

import numpy as np
import pandas as pd


class FeatureEngineerRefinado:
    def __init__(self):
        pass

    def _calcular_rsi(self, prices, period=14):
        """Calcula Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)  # evita divisão por zero
        return 100 - (100 / (1 + rs))

    def _calcular_stochastic(self, close, high, low, period=14):
        """Calcula Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)

    def _calcular_bollinger_upper(self, prices, period=20):
        """Bollinger Bands - Banda superior"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma + (std * 2)

    def _calcular_bollinger_lower(self, prices, period=20):
        """Bollinger Bands - Banda inferior"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma - (std * 2)

    def _calcular_atr(self, high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calcular_obv(self, close, volume):
        """On-Balance Volume"""
        returns = close.pct_change()
        obv = (volume * np.sign(returns)).cumsum()
        return obv

    def criar_features_basicas(self, df_ohlc: pd.DataFrame) -> pd.DataFrame:
        df = df_ohlc.copy()

        # Garantir colunas simples
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Features básicas
        df['retorno_1d'] = close.pct_change(1)
        df['retorno_3d'] = close.pct_change(3)
        df['retorno_5d'] = close.pct_change(5)

        # Médias móveis
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = close.rolling(window).mean()
            df[f'ema_{window}'] = close.ewm(span=window).mean()

        # Ratios entre médias
        df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
        df['sma_ratio_10_50'] = df['sma_10'] / df['sma_50']

        # Volatilidade
        for window in [5, 10, 20]:
            df[f'vol_{window}'] = close.pct_change().rolling(window).std()

        # Volume
        df['volume_ratio_5'] = volume / volume.rolling(5).mean()
        df['volume_ratio_10'] = volume / volume.rolling(10).mean()

        # Indicadores técnicos
        df['rsi_14'] = self._calcular_rsi(close, 14)
        df['stoch_14'] = self._calcular_stochastic(close, high, low, 14)

        # Bollinger Bands
        bb = self._calcular_bollinger_bands(close, 20)
        df['bollinger_upper'] = bb['upper']
        df['bollinger_lower'] = bb['lower']
        df['bollinger_pct'] = (close - bb['lower']) / (bb['upper'] - bb['lower'])

        # MACD
        df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR
        df['atr_14'] = self._calcular_atr(high, low, close, 14)

        # OBV
        df['obv'] = self._calcular_obv(close, volume)

        # CMF
        df['cmf_20'] = self._calcular_cmf(high, low, close, volume, 20)

        # Novas features adicionais
        df['high_low_ratio'] = high / low
        df['close_open_ratio'] = close / df['Open']
        df['volume_price_trend'] = volume * close.pct_change()

        # Novas features de volatilidade
        df['volatility_5'] = close.pct_change().rolling(5).std()
        df['volatility_20'] = close.pct_change().rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

        # Feature de tendência
        df['trend_strength'] = (close / close.rolling(20).mean() - 1) * 100

        # Feature de momentum
        df['momentum_5'] = close.pct_change(5)
        df['momentum_10'] = close.pct_change(10)

        return df.dropna()

    # ADICIONE ESTES MÉTODOS NOVOS na classe:
    def _calcular_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _calcular_stochastic(self, close, high, low, period=14):
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)

    def _calcular_bollinger_bands(self, prices, period=20):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return {'upper': sma + (std * 2), 'lower': sma - (std * 2)}

    def _calcular_atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calcular_obv(self, close, volume):
        returns = close.pct_change()
        obv = (volume * np.sign(returns.fillna(0))).cumsum()
        return obv

    def _calcular_cmf(self, high, low, close, volume, period=20):
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
        mf_volume = mf_multiplier * volume
        return mf_volume.rolling(period).sum() / volume.rolling(period).sum()

    def _calcular_adx(self, high, low, close, period=14):
        pass  # Implementação complexa - podemos adicionar depois

    def preparar_dataset_classificacao(self, df_ohlc: pd.DataFrame, periodo_historico: int = 504):
        """
        Prepara X (features), y (Series binária: 1 se subir, 0 se cair) e preços (Close).
        Retorna X, y, precos.
        """
        df_feat = self.criar_features_basicas(df_ohlc)
        close = df_feat['Close']

        # Label: direção do próximo dia
        future_close = close.shift(-1)
        y = (future_close > close).astype(int).dropna()

        if len(y) == 0:
            raise ValueError("Não há dados suficientes após o dropna() para criar labels")

        y_index = y.index

        # alinhar X com y
        X = df_feat.loc[y_index].drop(columns=['Open', 'High', 'Low', 'Close'], errors='ignore')
        precos = close.loc[y_index]

        # Agora podemos usar o 'y' original (a Series) ou convertê-lo
        y_final = y.reset_index(drop=True)
        y_final.name = 'target'

        return (X.reset_index(drop=True),
                y_final,
                precos.reset_index(drop=True))
