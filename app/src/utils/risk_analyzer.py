from typing import Dict, Any
import numpy as np
import pandas as pd
from src.config.params import Params
from src.logger.logger import logger
from src.utils.financial_calculation import CalculosEstatisticos


class RiskAnalyzer:
    """Realiza análise de risco e backtesting vetorial de estratégias."""

    def __init__(self, custo_por_trade_pct: float = None):
        self.custo_por_trade = custo_por_trade_pct or Params.CUSTO_POR_TRADE_PCT
        self.calculos = CalculosEstatisticos()

    def _retornar_metricas_vazias(self) -> Dict[str, Any]:
        """Retorna métricas padrão quando não há trades."""
        return {
            'retorno_total': 0.0, 'trades': 0, 'sharpe': 0.0,
            'max_drawdown': 0.0, 'equity_curve': [], 'win_rate': 0.0
        }

    def backtest_sinais(self, df_sinais: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """Executa um backtest vetorial com base nos sinais de trading."""
        if df_sinais.empty or 'sinal' not in df_sinais.columns or df_sinais['sinal'].sum() == 0:
            if verbose:
                logger.warning("Backtest não executado: sem sinais de operação.")
            return self._retornar_metricas_vazias()

        if not isinstance(df_sinais.index, pd.DatetimeIndex):
            df_sinais = df_sinais.copy()
            df_sinais.index = pd.to_datetime(df_sinais.index)

        df = df_sinais.copy()

        # Identifica mudanças de posição: 1 = entrar, -1 = sair
        df['posicao'] = df['sinal'].diff().fillna(0)

        trades = df[df['posicao'] != 0].copy()

        if trades.empty or trades['posicao'].iloc[0] == -1:
            return self._retornar_metricas_vazias()

        # Garante que a primeira operação é de entrada e a última de saída
        if trades['posicao'].iloc[-1] == 1:
            trades = trades.iloc[:-1]

        entradas = trades[trades['posicao'] == 1]['preco']
        saidas = trades[trades['posicao'] == -1]['preco']

        if len(entradas) > len(saidas):
            entradas = entradas.iloc[:len(saidas)]

        retornos = (saidas.values / entradas.values) - 1 - (self.custo_por_trade * 2)

        if len(retornos) == 0:
            return self._retornar_metricas_vazias()

        curva_equidade = np.cumprod(1 + retornos)

        metricas = {
            'retorno_total': float(curva_equidade[-1] - 1),
            'trades': len(retornos),
            'sharpe': self.calculos.calcular_sharpe_ratio(retornos),
            'max_drawdown': self.calculos.calcular_drawdown(np.insert(curva_equidade, 0, 1)),
            'win_rate': np.sum(retornos > 0) / len(retornos),
            'equity_curve': np.insert(curva_equidade, 0, 1).tolist()
        }

        if verbose:
            logger.info(
                f"Backtest: {metricas['trades']} trades, Retorno: {metricas['retorno_total']:.2%}, Sharpe: {metricas['sharpe']:.2f}")
        return metricas