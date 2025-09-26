from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src.config.params import Params
from src.logger.logger import logger
from src.models.technical_indicators import CalculosFinanceiros


class FeatureEngineer:
    """Realiza a engenharia de features e a criação de labels para os modelos de trading."""

    def __init__(self):
        """Inicializa o engenheiro de features com a classe de cálculos."""
        self.calculos = CalculosFinanceiros()

    def preparar_dataset(self, df_ohlc: pd.DataFrame, df_ibov: Optional[pd.DataFrame], ticker: str) -> Tuple[
        pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """Orquestra a preparação do dataset final com features e labels para o modelo."""
        logger.info("Iniciando criação de features e dataset...")

        # Etapa 1: Criar todas as features a partir dos dados brutos
        df_features = self._criar_features(df_ohlc, df_ibov)

        # Etapa 2: Limpar e preparar o dataframe de features
        df_features, X_untruncated = self._limpar_e_preparar_features(df_features)
        if df_features.empty:
            return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=float), pd.Series(dtype=object), pd.DataFrame()

        # Etapa 3: Criar labels com base nos preços originais e no dataframe de features
        labels, t1 = self._criar_labels_tripla_barreira(
            df_ohlc['Close'].loc[df_features.index],
            df_ohlc.loc[df_features.index],
            ticker
        )

        # Etapa 4: Alinhar features, labels e outros dados
        X, y, precos, t1 = self._alinhar_dados_finais(df_features, labels, df_ohlc['Close'], t1)

        logger.info(f"Dataset preparado - X: {X.shape}, y: {y.shape}")
        logger.info(f"Distribuição de labels: {y.value_counts(normalize=True).to_dict()}")

        return X, y, precos, t1, X_untruncated

    def _criar_features(self, df_ohlc: pd.DataFrame, df_ibov: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Cria um dataframe com todos os indicadores técnicos e de mercado."""
        df = df_ohlc.copy()
        df = self._adicionar_indicadores_tecnicos(df)
        df = self._adicionar_features_ibov(df, df_ibov)
        return df

    @staticmethod
    def _limpar_e_preparar_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove valores infinitos e nulos e prepara o dataframe para o labeling."""
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        X_untruncated = df.copy()
        return df, X_untruncated

    @staticmethod
    def _alinhar_dados_finais(X: pd.DataFrame, y: pd.Series, precos: pd.Series, t1: pd.Series) -> Tuple[
        pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Garante o alinhamento final entre todos os dataframes e series pelo índice."""
        t1 = t1.dropna()
        common_index = X.index.intersection(y.index).intersection(t1.index)
        X_final = X.loc[common_index]
        y_final = y.loc[common_index]
        precos_final = precos.loc[common_index]
        t1_final = t1.loc[common_index]

        # Remove colunas de dados brutos para evitar data leakage
        X_final.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore', inplace=True)
        return X_final, y_final, precos_final, t1_final

    def _adicionar_indicadores_tecnicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona um conjunto robusto de indicadores técnicos ao DataFrame, organizados por categoria."""
        df = self._adicionar_indicadores_momentum(df)
        df = self._adicionar_indicadores_tendencia(df)
        df = self._adicionar_indicadores_volatilidade(df)
        df = self._adicionar_indicadores_volume(df)
        return df

    def _adicionar_indicadores_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de Momentum."""
        close, high, low = df['Close'], df['High'], df['Low']
        df['rsi_14'] = self.calculos.calcular_rsi(close, 14)
        df['stoch_14'] = self.calculos.calcular_stochastic(close, high, low, 14)
        df['momentum_5d'] = close.pct_change(5)
        df['momentum_21d'] = close.pct_change(21)
        df['retorno_1d'] = close.pct_change(1)
        df['retorno_3d'] = close.pct_change(3)
        return df

    def _adicionar_indicadores_tendencia(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de Tendência."""
        close = df['Close']
        df['sma_ratio_10_50'] = self.calculos.calcular_medias_moveis(close, [10, 50])['sma_10'] / \
                                self.calculos.calcular_medias_moveis(close, [10, 50])['sma_50']
        df['sma_ratio_50_200'] = self.calculos.calcular_medias_moveis(close, [50, 200])['sma_50'] / \
                                 self.calculos.calcular_medias_moveis(close, [50, 200])['sma_200']
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df['macd_hist'] = macd - macd.ewm(span=9).mean()
        return df

    def _adicionar_indicadores_volatilidade(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de Volatilidade."""
        close, high, low = df['Close'], df['High'], df['Low']
        df['bollinger_pct'] = self.calculos.calcular_bandas_bollinger(close, 20)['pct_b']
        df['atr_14_norm'] = self.calculos.calcular_atr(high, low, close, 14) / close
        df['vol_21d'] = close.pct_change().rolling(21).std() * np.sqrt(252)
        df['vol_of_vol_10d'] = df['vol_21d'].rolling(10).std()
        return df

    def _adicionar_indicadores_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de Volume."""
        close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
        df['cmf_20'] = self.calculos.calcular_cmf(high, low, close, volume, 20)
        df['volume_ratio_21d'] = volume / volume.rolling(21).mean()
        obv = self.calculos.calcular_obv(close, volume)
        df['obv_norm_21d'] = (obv - obv.rolling(21).min()) / (obv.rolling(21).max() - obv.rolling(21).min() + 1e-9)
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

    @staticmethod
    def _get_event_end_time(precos: pd.Series, t0: pd.Timestamp, pt: float, sl: float,
                            n_dias: int) -> pd.Timestamp:
        """
        Encontra o timestamp em que uma das barreiras (lucro, perda ou tempo) é atingida.
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