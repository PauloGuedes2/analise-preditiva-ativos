from typing import Dict

import numpy as np
import pandas as pd


class CalculosFinanceiros:
    """Classe utilitária com métodos estáticos para cálculos de indicadores financeiros."""

    @staticmethod
    def calcular_rsi(precos: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula o Relative Strength Index (RSI)."""
        delta = precos.diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganho / (perda + 1e-9)  # Adiciona epsilon para evitar divisão por zero
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calcular_stochastic(fechamento: pd.Series, alta: pd.Series,
                            baixa: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula o Stochastic Oscillator."""
        menor_baixa = baixa.rolling(window=periodo).min()
        maior_alta = alta.rolling(window=periodo).max()
        denominador = maior_alta - menor_baixa
        denominador = denominador.replace(0, np.nan)  # Evita divisão por zero
        return 100 * (fechamento - menor_baixa) / denominador

    @staticmethod
    def calcular_bandas_bollinger(precos: pd.Series, periodo: int = 20) -> Dict[str, pd.Series]:
        """Calcula as Bandas de Bollinger e o indicador %B."""
        media_movel = precos.rolling(window=periodo).mean()
        desvio_padrao = precos.rolling(window=periodo).std()

        banda_superior = media_movel + (desvio_padrao * 2)
        banda_inferior = media_movel - (desvio_padrao * 2)

        # %B indica a posição do preço em relação às bandas
        pct_b = (precos - banda_inferior) / (banda_superior - banda_inferior + 1e-9)

        return {
            'superior': banda_superior,
            'inferior': banda_inferior,
            'pct_b': pct_b
        }

    @staticmethod
    def calcular_atr(alta: pd.Series, baixa: pd.Series,
                     fechamento: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula o Average True Range (ATR)."""
        tr1 = alta - baixa
        tr2 = abs(alta - fechamento.shift())
        tr3 = abs(baixa - fechamento.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(periodo).mean()

    @staticmethod
    def calcular_obv(fechamento: pd.Series, volume: pd.Series) -> pd.Series:
        """Calcula o On-Balance Volume (OBV)."""
        retornos = fechamento.pct_change()
        # Acumula o volume com base na direção do preço
        return (volume * np.sign(retornos.fillna(0))).cumsum()

    @staticmethod
    def calcular_cmf(alta: pd.Series, baixa: pd.Series, fechamento: pd.Series,
                     volume: pd.Series, periodo: int = 20) -> pd.Series:
        """Calcula o Chaikin Money Flow (CMF)."""
        multiplicador_mf = ((fechamento - baixa) - (alta - fechamento)) / (alta - baixa + 1e-9)
        volume_mf = multiplicador_mf * volume
        return volume_mf.rolling(periodo).sum() / volume.rolling(periodo).sum()

    @staticmethod
    def calcular_medias_moveis(precos: pd.Series, janelas: list = None) -> Dict[str, pd.Series]:
        """Calcula médias móveis simples."""
        if janelas is None:
            janelas = [5, 10, 20, 50, 100]
        medias = {}
        for janela in janelas:
            medias[f'sma_{janela}'] = precos.rolling(janela).mean()
        return medias


class CalculosEstatisticos:
    """Classe com métodos para cálculos estatísticos."""

    @staticmethod
    def calcular_sharpe_ratio(retornos: np.ndarray, dias_anuais: int = 252) -> float:
        """Calcula o Sharpe Ratio anualizado."""
        if len(retornos) < 2 or np.std(retornos) == 0:
            return 0.0
        return (np.mean(retornos) / np.std(retornos)) * np.sqrt(dias_anuais)

    @staticmethod
    def calcular_sortino_ratio(retornos: np.ndarray, dias_anuais: int = 252) -> float:
        """Calcula o Sortino Ratio anualizado, focando no risco de perdas."""
        if len(retornos) < 2:
            return 0.0

        retornos_negativos = retornos[retornos < 0]
        if len(retornos_negativos) < 2:
            return np.inf if np.mean(retornos) > 0 else 0.0

        downside_std = np.std(retornos_negativos)
        if downside_std == 0:
            return np.inf if np.mean(retornos) > 0 else 0.0

        sortino = (np.mean(retornos) / downside_std) * np.sqrt(dias_anuais)
        return float(sortino)
