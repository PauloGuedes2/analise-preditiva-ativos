from typing import Dict

import numpy as np
import pandas as pd

from src.config.params import Params


class CalculosFinanceiros:
    """Classe com métodos para cálculos financeiros."""

    @staticmethod
    def calcular_rsi(precos: pd.Series, periodo: int = None) -> pd.Series:
        """Calcula o Relative Strength Index (RSI)."""
        if periodo is None:
            periodo = Params.INDICADORES_CONFIG['rsi_periodo']

        delta = precos.diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganho / (perda + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calcular_stochastic(fechamento: pd.Series, alta: pd.Series,
                            baixa: pd.Series, periodo: int = None) -> pd.Series:
        """Calcula o Stochastic Oscillator."""
        if periodo is None:
            periodo = Params.INDICADORES_CONFIG['stoch_periodo']

        menor_baixa = baixa.rolling(window=periodo).min()
        maior_alta = alta.rolling(window=periodo).max()
        return 100 * (fechamento - menor_baixa) / (maior_alta - menor_baixa + 1e-9)

    @staticmethod
    def calcular_bandas_bollinger(precos: pd.Series, periodo: int = None) -> Dict[str, pd.Series]:
        """Calcula as Bandas de Bollinger."""
        if periodo is None:
            periodo = Params.INDICADORES_CONFIG['bollinger_periodo']

        media_movel = precos.rolling(window=periodo).mean()
        desvio_padrao = precos.rolling(window=periodo).std()

        return {
            'superior': media_movel + (desvio_padrao * 2),
            'inferior': media_movel - (desvio_padrao * 2)
        }

    @staticmethod
    def calcular_atr(alta: pd.Series, baixa: pd.Series,
                     fechamento: pd.Series, periodo: int = None) -> pd.Series:
        """Calcula o Average True Range (ATR)."""
        if periodo is None:
            periodo = Params.INDICADORES_CONFIG['atr_periodo']

        tr1 = alta - baixa
        tr2 = abs(alta - fechamento.shift())
        tr3 = abs(baixa - fechamento.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(periodo).mean()

    @staticmethod
    def calcular_obv(fechamento: pd.Series, volume: pd.Series) -> pd.Series:
        """Calcula o On-Balance Volume (OBV)."""
        retornos = fechamento.pct_change()
        return (volume * np.sign(retornos.fillna(0))).cumsum()

    @staticmethod
    def calcular_cmf(alta: pd.Series, baixa: pd.Series, fechamento: pd.Series,
                     volume: pd.Series, periodo: int = None) -> pd.Series:
        """Calcula o Chaikin Money Flow (CMF)."""
        if periodo is None:
            periodo = Params.INDICADORES_CONFIG['cmf_periodo']

        multiplicador_mf = ((fechamento - baixa) - (alta - fechamento)) / (alta - baixa + 1e-9)
        volume_mf = multiplicador_mf * volume
        return volume_mf.rolling(periodo).sum() / volume.rolling(periodo).sum()

    @staticmethod
    def calcular_retornos(precos: pd.Series, periodos: list = None) -> Dict[str, pd.Series]:
        """Calcula retornos para múltiplos períodos."""
        if periodos is None:
            periodos = Params.JANELAS_RETORNOS

        retornos = {}
        for periodo in periodos:
            retornos[f'retorno_{periodo}d'] = precos.pct_change(periodo)
        return retornos

    @staticmethod
    def calcular_medias_moveis(precos: pd.Series, janelas: list = None) -> Dict[str, pd.Series]:
        """Calcula médias móveis simples e exponenciais."""
        if janelas is None:
            janelas = Params.JANELAS_MEDIAS_MOVEIS

        medias = {}
        for janela in janelas:
            medias[f'sma_{janela}'] = precos.rolling(janela).mean()
            medias[f'ema_{janela}'] = precos.ewm(span=janela).mean()
        return medias


class CalculosEstatisticos:
    """Classe com métodos para cálculos estatísticos."""

    @staticmethod
    def calcular_sharpe_ratio(retornos: np.ndarray, dias_anuais: int = 252) -> float:
        """Calcula o Sharpe Ratio anualizado."""
        if len(retornos) == 0 or np.std(retornos) == 0:
            return 0.0
        return (np.mean(retornos) / np.std(retornos)) * np.sqrt(dias_anuais)

    @staticmethod
    def calcular_drawdown(curva_equidade: np.ndarray) -> float:
        """Calcula o máximo drawdown."""
        if len(curva_equidade) == 0:
            return 0.0

        pico = np.maximum.accumulate(curva_equidade)
        drawdowns = (curva_equidade - pico) / pico
        return float(np.min(drawdowns))

    @staticmethod
    def calcular_var(retornos: np.ndarray, nivel_confianca: float = None) -> float:
        """Calcula Value at Risk (VaR) histórico."""
        if nivel_confianca is None:
            nivel_confianca = Params.NIVEL_CONFIANCA_VAR

        if len(retornos) == 0:
            return 0.0

        return float(np.percentile(retornos, (1 - nivel_confianca) * 100))

    @staticmethod
    def calcular_cvar(retornos: np.ndarray, nivel_confianca: float = None) -> float:
        """Calcula Conditional Value at Risk (CVaR)."""
        if nivel_confianca is None:
            nivel_confianca = Params.NIVEL_CONFIANCA_VAR

        if len(retornos) == 0:
            return 0.0

        var = CalculosEstatisticos.calcular_var(retornos, nivel_confianca)
        retornos_abaixo_var = retornos[retornos <= var]

        if len(retornos_abaixo_var) == 0:
            return var

        return float(np.mean(retornos_abaixo_var))

    @staticmethod
    def calcular_correlacao_rolling(serie1: pd.Series, serie2: pd.Series,
                                    janela: int = 20) -> pd.Series:
        """Calcula correlação rolling entre duas séries."""
        return serie1.rolling(janela).corr(serie2)
