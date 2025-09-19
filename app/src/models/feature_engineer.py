from typing import Tuple, Optional

import pandas as pd

from src.logger.logger import logger
from src.utils.financial_calculation import CalculosFinanceiros


class FeatureEngineer:
    """Realiza engenharia de features para dados financeiros."""

    def __init__(self):
        self.calculos = CalculosFinanceiros()

    def _adicionar_features_basicas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features básicas ao DataFrame."""
        fechamento = df['Close']

        # Retornos
        retornos = self.calculos.calcular_retornos(fechamento)
        for nome, valor in retornos.items():
            df[nome] = valor

        # Médias móveis
        medias = self.calculos.calcular_medias_moveis(fechamento)
        for nome, valor in medias.items():
            df[nome] = valor

        # Ratios entre médias
        df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
        df['sma_ratio_10_50'] = df['sma_10'] / df['sma_50']

        return df

    def _adicionar_indicadores_tecnicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos ao DataFrame."""
        fechamento = df['Close']
        alta = df['High']
        baixa = df['Low']
        volume = df['Volume']

        # RSI e Stochastic
        df['rsi_14'] = self.calculos.calcular_rsi(fechamento)
        df['stoch_14'] = self.calculos.calcular_stochastic(fechamento, alta, baixa)

        # Bandas de Bollinger
        bandas = self.calculos.calcular_bandas_bollinger(fechamento)
        df['bollinger_upper'] = bandas['superior']
        df['bollinger_lower'] = bandas['inferior']
        df['bollinger_pct'] = (fechamento - bandas['inferior']) / (bandas['superior'] - bandas['inferior'] + 1e-9)

        # MACD
        df['macd'] = fechamento.ewm(span=12).mean() - fechamento.ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR e OBV
        df['atr_14'] = self.calculos.calcular_atr(alta, baixa, fechamento)
        df['obv'] = self.calculos.calcular_obv(fechamento, volume)

        return df

    @staticmethod
    def _adicionar_features_avancadas(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features avançadas ao DataFrame."""
        fechamento = df['Close']
        alta = df['High']
        baixa = df['Low']
        volume = df['Volume']

        # Ratios de preço
        df['high_low_ratio'] = alta / baixa
        df['close_open_ratio'] = fechamento / df['Open']
        df['volume_price_trend'] = volume * fechamento.pct_change()

        # Volatilidade
        df['volatility_5'] = fechamento.pct_change().rolling(5).std()
        df['volatility_20'] = fechamento.pct_change().rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

        # Tendência e momentum
        df['trend_strength'] = (fechamento / fechamento.rolling(20).mean() - 1) * 100
        df['momentum_5'] = fechamento.pct_change(5)
        df['momentum_10'] = fechamento.pct_change(10)

        return df

    @staticmethod
    def _adicionar_features_ibov(df: pd.DataFrame, df_ibov: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Adiciona features relacionadas ao IBOV."""
        if df_ibov is not None:
            df_ibov_alinhado = df_ibov.reindex(df.index).ffill()

            df['retorno_1d_ibov'] = df_ibov_alinhado['Close_IBOV'].pct_change(1)
            df['forca_relativa_ibov'] = df['retorno_1d'] - df['retorno_1d_ibov']
            df['correlacao_ibov_20d'] = df['retorno_1d'].rolling(20).corr(df['retorno_1d_ibov'])
            df['sma_50_ibov'] = df_ibov_alinhado['Close_IBOV'].rolling(50).mean()
            df['ibov_acima_sma50'] = (df_ibov_alinhado['Close_IBOV'] > df['sma_50_ibov']).astype(int)

        return df

    def criar_features(self, df_ohlc: pd.DataFrame, df_ibov: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Cria features completas a partir de dados OHLC.
        """
        df = df_ohlc.copy()

        # Garantir colunas simples
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Pipeline de criação de features
        df = self._adicionar_features_basicas(df)
        df = self._adicionar_indicadores_tecnicos(df)
        df = self._adicionar_features_avancadas(df)
        df = self._adicionar_features_ibov(df, df_ibov)

        logger.info(f"Features criadas - Total: {len(df.columns)} colunas")
        return df.dropna()

    def preparar_dataset_classificacao(self, df_ohlc: pd.DataFrame,
                                       df_ibov: Optional[pd.DataFrame] = None) -> Tuple[
        pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dataset para classificação.
        """
        df_features = self.criar_features(df_ohlc, df_ibov)
        fechamento = df_features['Close']

        # Criar labels: 1 se subir, 0 se cair
        fechamento_futuro = fechamento.shift(-1)
        y = (fechamento_futuro > fechamento).astype(int).dropna()

        if y.empty:
            raise ValueError("Dados insuficientes para criar labels após dropna()")

        # Alinhar features com labels
        X = df_features.loc[y.index].drop(
            columns=['Open', 'High', 'Low', 'Close'],
            errors='ignore'
        )

        precos = fechamento.loc[y.index]

        logger.info(f"Dataset preparado - X: {X.shape}, y: {y.shape}")
        return (
            X.reset_index(drop=True),
            y.reset_index(drop=True).rename('target'),
            precos.reset_index(drop=True)
        )
