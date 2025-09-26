from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src.config.params import Params
from src.logger.logger import logger
from src.models.technical_indicators import CalculosFinanceiros


class FeatureEngineer:
    """Realiza a engenharia de features e a criação de labels para os modelos de trading."""

    def __init__(self):
        self.calculos = CalculosFinanceiros()

    def _adicionar_indicadores_tecnicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona um conjunto robusto de indicadores técnicos ao DataFrame."""
        close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

        # Indicadores de Momentum
        df['rsi_14'] = self.calculos.calcular_rsi(close, 14)
        df['stoch_14'] = self.calculos.calcular_stochastic(close, high, low, 14)
        df['momentum_5d'] = close.pct_change(5)
        df['momentum_21d'] = close.pct_change(21)

        # Indicadores de Tendência
        df['sma_ratio_10_50'] = self.calculos.calcular_medias_moveis(close, [10, 50])['sma_10'] / \
                                self.calculos.calcular_medias_moveis(close, [10, 50])['sma_50']

        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df['macd_hist'] = macd - macd.ewm(span=9).mean()

        # Indicadores de Volatilidade
        df['bollinger_pct'] = self.calculos.calcular_bandas_bollinger(close, 20)['pct_b']
        df['atr_14_norm'] = self.calculos.calcular_atr(high, low, close, 14) / close
        df['vol_21d'] = close.pct_change().rolling(21).std() * np.sqrt(252)

        # Indicadores de Volume
        df['cmf_20'] = self.calculos.calcular_cmf(high, low, close, volume, 20)
        df['volume_ratio_21d'] = volume / volume.rolling(21).mean()

        # Indicadores de Volatilidade Avançada
        df['vol_of_vol_10d'] = df['vol_21d'].rolling(10).std()

        # Indicador de Tendência de Longo Prazo
        df['sma_ratio_50_200'] = self.calculos.calcular_medias_moveis(close, [50, 200])['sma_50'] / \
                                 self.calculos.calcular_medias_moveis(close, [50, 200])['sma_200']

        # Indicador de Volume Acumulado
        obv = self.calculos.calcular_obv(close, volume)
        df['obv_norm_21d'] = (obv - obv.rolling(21).min()) / (obv.rolling(21).max() - obv.rolling(21).min() + 1e-9)

        # Indicador de Retornos Simples de Curto Prazo
        df['retorno_1d'] = close.pct_change(1)
        df['retorno_3d'] = close.pct_change(3)

        return df

    @staticmethod
    def _adicionar_features_ibov(df: pd.DataFrame, df_ibov: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Adiciona features de mercado (IBOV) com tratamento robusto de dados."""
        if df_ibov is not None and not df_ibov.empty and 'Close_IBOV' in df_ibov.columns:
            try:
                # Alinha o índice do IBOV com o do ativo, preenchendo dias faltantes
                df_ibov_alinhado = df_ibov.reindex(df.index).ffill().bfill()

                # Adicionar feature de correlação de 20 dias com o IBOV
                if len(df_ibov_alinhado) > 20:
                    correlacao = df['Close'].pct_change().rolling(20).corr(
                        df_ibov_alinhado['Close_IBOV'].pct_change())
                    df['correlacao_ibov_20d'] = correlacao

                # Adicionar feature de posição relativa ao SMA de 50 dias do IBOV
                if len(df_ibov_alinhado) > 50:
                    df['ibov_acima_sma50'] = (
                            df_ibov_alinhado['Close_IBOV'] >
                            df_ibov_alinhado['Close_IBOV'].rolling(50).mean()
                    ).astype(int)

            except Exception as e:
                logger.warning(f"Erro ao processar features do IBOV: {e}")

        return df

    @staticmethod
    def _get_event_end_time(precos: pd.Series, t0: pd.Timestamp, pt: float, sl: float,
                            n_dias: int) -> pd.Timestamp:
        """
        Encontra o timestamp em que uma das barreiras (lucro, perda ou tempo) é atingida.

        Args:
            precos (pd.Series): Série de preços futuros.
            t0 (pd.Timestamp): Timestamp de início do evento.
            pt (float): Preço da barreira de lucro (profit taking).
            sl (float): Preço da barreira de perda (stop loss).
            n_dias (int): Número máximo de dias para o evento.

        Returns:
            pd.Timestamp: O timestamp de término do evento.
        """
        janela = precos[t0:].iloc[1:n_dias + 1]

        atingiu_superior = janela[janela >= pt]
        atingiu_inferior = janela[janela <= sl]

        # Retorna o tempo do primeiro toque em qualquer barreira
        if not atingiu_inferior.empty and not atingiu_superior.empty:
            return min(atingiu_superior.index[0], atingiu_inferior.index[0])
        elif not atingiu_superior.empty:
            return atingiu_superior.index[0]
        elif not atingiu_inferior.empty:
            return atingiu_inferior.index[0]

        # Retorna o final da janela se nenhuma barreira for tocada
        return janela.index[-1] if not janela.empty else t0 + pd.Timedelta(days=n_dias)

    def _criar_labels_tripla_barreira(
            self,
            precos: pd.Series,
            df_completo: pd.DataFrame,
            ticker: str,
            lookahead_days: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """"Cria labels usando a metodologia de tripla barreira com volatilidade adaptativa baseada no ATR"""
        n_dias = lookahead_days or Params.TRIPLE_BARRIER_LOOKAHEAD_DAYS

        if len(precos) < n_dias * 2:
            logger.warning(f"Dados insuficientes para tripla barreira em {ticker}")
            return pd.Series(dtype=int), pd.Series(dtype=object)

        volatilidade = self.calculos.calcular_atr(df_completo['High'], df_completo['Low'], df_completo['Close'], 14)
        volatilidade = volatilidade.reindex(precos.index).ffill().bfill()

        fator_pt, fator_sl = Params.ATR_FACTORS.get(ticker, Params.ATR_FACTORS["DEFAULT"])

        barreira_superior = precos + (fator_pt * volatilidade)
        barreira_inferior = precos - (fator_sl * volatilidade)

        labels = pd.Series(0, index=precos.index)
        t1 = pd.Series(pd.NaT, index=precos.index)

        n = min(len(precos) - n_dias, len(precos) - 1)
        for i in range(n):
            t0 = precos.index[i]
            pt = barreira_superior.iloc[i]
            sl = barreira_inferior.iloc[i]
            t1.iloc[i] = self._get_event_end_time(precos, t0, pt, sl, n_dias)

            event_prices = precos[t0:t1.iloc[i]]

            if not event_prices.empty:
                atingiu_superior = (event_prices >= pt).any()
                atingiu_inferior = (event_prices <= sl).any()

                if atingiu_superior:
                    labels.iloc[i] = 1
                elif atingiu_inferior:
                    labels.iloc[i] = -1

        return labels, t1

    def preparar_dataset(self, df_ohlc: pd.DataFrame, df_ibov: Optional[pd.DataFrame], ticker: str) -> Tuple[
        pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """Prepara o dataset final com features e labels para o modelo"""
        logger.info("Iniciando criação de features e dataset...")

        df = df_ohlc.copy()

        # Adicionar features
        df = self._adicionar_indicadores_tecnicos(df)
        df = self._adicionar_features_ibov(df, df_ibov)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        X_untruncated = df.copy()

        if df.empty:
            return (pd.DataFrame(),
                    pd.Series(dtype=int),
                    pd.Series(dtype=float),
                    pd.Series(dtype=object),
                    pd.DataFrame())

        # Criar labels usando os preços originais
        labels, t1 = self._criar_labels_tripla_barreira(
            df_ohlc['Close'].loc[df.index],
            df_ohlc.loc[df.index],
            ticker,
            lookahead_days=Params.TRIPLE_BARRIER_LOOKAHEAD_DAYS
        )

        # Alinhar X e y
        X = df.loc[labels.index]
        y = labels
        precos = df_ohlc['Close'].loc[labels.index]
        t1 = t1.dropna()

        # Garantir alinhamento final
        common_index = X.index.intersection(y.index).intersection(t1.index)
        X, y, precos, t1 = X.loc[common_index], y.loc[common_index], precos.loc[common_index], t1.loc[common_index]

        X.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore', inplace=True)

        logger.info(f"Dataset preparado - X: {X.shape}, y: {y.shape}")
        logger.info(f"Distribuição de labels: {y.value_counts(normalize=True).to_dict()}")

        return X, y, precos, t1, X_untruncated
