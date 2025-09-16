from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Realiza engenharia de features para dados financeiros."""

    def __init__(self):
        self._indicadores_config = {
            'rsi_periodo': 14,
            'stoch_periodo': 14,
            'bollinger_periodo': 20,
            'atr_periodo': 14,
            'cmf_periodo': 20
        }

    @staticmethod
    def _calcular_indicador_rsi(precos: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula o Relative Strength Index (RSI)."""
        delta = precos.diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganho / (perda + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calcular_indicador_stochastic(fechamento: pd.Series, alta: pd.Series,
                                       baixa: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula o Stochastic Oscillator."""
        menor_baixa = baixa.rolling(window=periodo).min()
        maior_alta = alta.rolling(window=periodo).max()
        return 100 * (fechamento - menor_baixa) / (maior_alta - menor_baixa + 1e-9)

    @staticmethod
    def _calcular_bandas_bollinger(precos: pd.Series, periodo: int = 20) -> Dict[str, pd.Series]:
        """Calcula as Bandas de Bollinger."""
        media_movel = precos.rolling(window=periodo).mean()
        desvio_padrao = precos.rolling(window=periodo).std()

        return {
            'superior': media_movel + (desvio_padrao * 2),
            'inferior': media_movel - (desvio_padrao * 2)
        }

    @staticmethod
    def _calcular_atr(alta: pd.Series, baixa: pd.Series,
                      fechamento: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula o Average True Range (ATR)."""
        tr1 = alta - baixa
        tr2 = abs(alta - fechamento.shift())
        tr3 = abs(baixa - fechamento.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(periodo).mean()

    @staticmethod
    def _calcular_obv(fechamento: pd.Series, volume: pd.Series) -> pd.Series:
        """Calcula o On-Balance Volume (OBV)."""
        retornos = fechamento.pct_change()
        return (volume * np.sign(retornos.fillna(0))).cumsum()

    @staticmethod
    def _calcular_cmf(alta: pd.Series, baixa: pd.Series, fechamento: pd.Series,
                      volume: pd.Series, periodo: int = 20) -> pd.Series:
        """Calcula o Chaikin Money Flow (CMF)."""
        multiplicador_mf = ((fechamento - baixa) - (alta - fechamento)) / (alta - baixa + 1e-9)
        volume_mf = multiplicador_mf * volume
        return volume_mf.rolling(periodo).sum() / volume.rolling(periodo).sum()

    @staticmethod
    def _adicionar_features_basicas(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features básicas ao DataFrame."""
        fechamento = df['Close']

        # Retornos
        for dias in [1, 3, 5]:
            df[f'retorno_{dias}d'] = fechamento.pct_change(dias)

        # Médias móveis
        for janela in [5, 10, 20, 50]:
            df[f'sma_{janela}'] = fechamento.rolling(janela).mean()
            df[f'ema_{janela}'] = fechamento.ewm(span=janela).mean()

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
        df['rsi_14'] = self._calcular_indicador_rsi(fechamento, 14)
        df['stoch_14'] = self._calcular_indicador_stochastic(fechamento, alta, baixa, 14)

        # Bandas de Bollinger
        bandas = self._calcular_bandas_bollinger(fechamento, 20)
        df['bollinger_upper'] = bandas['superior']
        df['bollinger_lower'] = bandas['inferior']
        df['bollinger_pct'] = (fechamento - bandas['inferior']) / (bandas['superior'] - bandas['inferior'] + 1e-9)

        # MACD
        df['macd'] = fechamento.ewm(span=12).mean() - fechamento.ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR e OBV
        df['atr_14'] = self._calcular_atr(alta, baixa, fechamento, 14)
        df['obv'] = self._calcular_obv(fechamento, volume)

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

        Args:
            df_ohlc: DataFrame com dados OHLC
            df_ibov: DataFrame com dados do IBOV (opcional)

        Returns:
            DataFrame com features engineering
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

        return df.dropna()

    def preparar_dataset_classificacao(self, df_ohlc: pd.DataFrame,
                                       df_ibov: Optional[pd.DataFrame] = None) -> Tuple[
        pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dataset para classificação.

        Returns:
            Tuple com (X_features, y_target, preços)
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

        return (
            X.reset_index(drop=True),
            y.reset_index(drop=True).rename('target'),
            precos.reset_index(drop=True)
        )
