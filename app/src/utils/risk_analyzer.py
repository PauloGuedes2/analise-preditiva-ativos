from typing import Dict

import numpy as np
import pandas as pd

from src.config.params import Params
from src.logger.logger import logger
from src.utils.financial_calculation import CalculosEstatisticos


class RiskAnalyzer:
    """Realiza análise de risco e backtesting de estratégias."""

    def __init__(self, custo_por_trade_pct: float = None):
        self.custo_por_trade_pct = custo_por_trade_pct or Params.CUSTO_POR_TRADE_PCT
        self.calculos = CalculosEstatisticos()

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

    def backtest_sinais(self, df_sinais: pd.DataFrame) -> Dict[str, float]:
        """
        Executa backtest com base em sinais de trading.
        """
        if len(df_sinais) < 2:
            return self._retornar_metricas_vazias()

        df = df_sinais.reset_index(drop=True).copy()
        retornos = self._calcular_retornos(df)

        if len(retornos) == 0:
            return self._retornar_metricas_vazias()

        curva_equidade = np.cumprod(1 + retornos)
        max_drawdown = self.calculos.calcular_drawdown(curva_equidade)
        sharpe_ratio = self.calculos.calcular_sharpe_ratio(retornos)

        metricas = {
            'retorno_total': float(np.prod(1 + retornos) - 1),
            'trades': int(len(retornos)),
            'sharpe': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'retorno_med_diario': float(np.mean(retornos)),
            'equity_curve': curva_equidade.tolist()
        }

        logger.info(f"Backtest realizado - Trades: {metricas['trades']}, "
                    f"Retorno: {metricas['retorno_total']:.2%}, "
                    f"Sharpe: {metricas['sharpe']:.2f}")

        return metricas

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

    def calcular_var(self, retornos: np.ndarray, nivel_confianca: float = None) -> float:
        """Calcula Value at Risk (VaR) histórico."""
        return self.calculos.calcular_var(retornos, nivel_confianca)

    def calcular_cvar(self, retornos: np.ndarray, nivel_confianca: float = None) -> float:
        """Calcula Conditional Value at Risk (CVaR)."""
        return self.calculos.calcular_cvar(retornos, nivel_confianca)
