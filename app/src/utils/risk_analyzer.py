from typing import Dict

import numpy as np
import pandas as pd


class RiskAnalyzer:
    """Realiza análise de risco e backtesting de estratégias."""

    def __init__(self, custo_por_trade_pct: float = 0.0005):
        self.custo_por_trade_pct = custo_por_trade_pct

    def _calcular_retornos(self, df_sinais: pd.DataFrame) -> np.ndarray:
        """Calcula retornos individuais dos trades."""
        retornos = []
        n = len(df_sinais)

        for i in range(n - 1):
            if df_sinais.loc[i, 'pred'] == 1:
                preco_entrada = df_sinais.loc[i, 'preco']
                preco_saida = df_sinais.loc[i + 1, 'preco']

                retorno = (preco_saida / preco_entrada) - 1.0
                retorno -= 2 * self.custo_por_trade_pct  # Custos de entrada e saída

                retornos.append(retorno)

        return np.array(retornos)

    @staticmethod
    def _calcular_curva_equidade(retornos: np.ndarray) -> np.ndarray:
        """Calcula a curva de equidade."""
        return np.cumprod(1 + retornos)

    @staticmethod
    def _calcular_drawdown(curva_equidade: np.ndarray) -> float:
        """Calcula o máximo drawdown."""
        pico = np.maximum.accumulate(curva_equidade)
        drawdowns = (curva_equidade - pico) / pico
        return float(np.min(drawdowns))

    @staticmethod
    def _calcular_sharpe_ratio(retornos: np.ndarray) -> float:
        """Calcula o Sharpe Ratio anualizado."""
        if len(retornos) == 0 or np.std(retornos) == 0:
            return 0.0

        return (np.mean(retornos) / np.std(retornos)) * np.sqrt(252)

    def backtest_sinais(self, df_sinais: pd.DataFrame) -> Dict[str, float]:
        """
        Executa backtest com base em sinais de trading.

        Args:
            df_sinais: DataFrame com colunas ['preco', 'proba', 'pred']

        Returns:
            Dicionário com métricas de performance
        """
        if len(df_sinais) < 2:
            return self._retornar_metricas_vazias()

        df = df_sinais.reset_index(drop=True).copy()
        retornos = self._calcular_retornos(df)

        if len(retornos) == 0:
            return self._retornar_metricas_vazias()

        curva_equidade = self._calcular_curva_equidade(retornos)
        max_drawdown = self._calcular_drawdown(curva_equidade)
        sharpe_ratio = self._calcular_sharpe_ratio(retornos)

        return {
            'retorno_total': float(np.prod(1 + retornos) - 1),
            'trades': int(len(retornos)),
            'sharpe': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'retorno_med_diario': float(np.mean(retornos)),
            'equity_curve': curva_equidade.tolist()
        }

    @staticmethod
    def _retornar_metricas_vazias() -> Dict[str, float]:
        """Retorna métricas padrão para casos sem trades."""
        return {
            'retorno_total': 0.0,
            'trades': 0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'retorno_med_diario': 0.0,
            'equity_curve': []
        }

    @staticmethod
    def calcular_var(retornos: np.ndarray, nivel_confianca: float = 0.95) -> float:
        """
        Calcula Value at Risk (VaR) histórico.

        Args:
            retornos: Array de retornos
            nivel_confianca: Nível de confiança para VaR

        Returns:
            VaR no nível de confiança especificado
        """
        if len(retornos) == 0:
            return 0.0

        return float(np.percentile(retornos, (1 - nivel_confianca) * 100))

    def calcular_cvar(self, retornos: np.ndarray, nivel_confianca: float = 0.95) -> float:
        """
        Calcula Conditional Value at Risk (CVaR).

        Args:
            retornos: Array de retornos
            nivel_confianca: Nível de confiança para CVaR

        Returns:
            CVaR no nível de confiança especificado
        """
        if len(retornos) == 0:
            return 0.0

        var = self.calcular_var(retornos, nivel_confianca)
        retornos_abaixo_var = retornos[retornos <= var]

        if len(retornos_abaixo_var) == 0:
            return var

        return float(np.mean(retornos_abaixo_var))
