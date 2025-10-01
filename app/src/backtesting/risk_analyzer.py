from typing import Dict, Any
import numpy as np
import pandas as pd
from src.config.params import Params
from src.logger.logger import logger
from src.models.technical_indicators import CalculosEstatisticos


class RiskAnalyzer:
    """Realiza análise de risco e backtesting vetorial de estratégias."""

    def __init__(self, custo_por_trade_pct: float = None):
        """
        Inicializa o analisador de risco.

        Args:
            custo_por_trade_pct (float, optional): Custo percentual por operação.
                                                   Se não fornecido, usa o padrão de `Params`.
        """
        self.custo_por_trade = custo_por_trade_pct or Params.CUSTO_POR_TRADE_PCT
        self.calculos = CalculosEstatisticos()

    @staticmethod
    def retornar_metricas_vazias() -> Dict[str, Any]:
        """Retorna um dicionário com métricas zeradas para casos sem operações."""
        return {
            'retorno_total': 0.0, 'trades': 0, 'sharpe': 0.0, 'sortino': 0.0,
            'max_drawdown': 0.0, 'equity_curve': [], 'win_rate': 0.0,
            'profit_factor': 0.0, 'payoff_ratio': 0.0, 'drawdown_series': []
        }

    def backtest_sinais(self, df_sinais: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """
        Executa um backtest vetorial a partir de um DataFrame de sinais.

        Args:
            df_sinais (pd.DataFrame): DataFrame com colunas 'preco' e 'sinal' (1 para comprar/manter, 0 para vender/ficar fora).
            verbose (bool, optional): Se True, loga um resumo do resultado.

        Returns:
            Dict[str, Any]: Dicionário com diversas métricas de performance da estratégia.
        """
        if df_sinais.empty or 'sinal' not in df_sinais.columns or df_sinais['sinal'].sum() == 0:
            if verbose: logger.warning("Backtest não executado: sem sinais de operação.")
            return self.retornar_metricas_vazias()

        if not isinstance(df_sinais.index, pd.DatetimeIndex):
            df_sinais = df_sinais.copy()
            df_sinais.index = pd.to_datetime(df_sinais.index)

        df = df_sinais.copy()
        # 'posicao' marca as mudanças de sinal (entradas e saídas)
        df['posicao'] = df['sinal'].diff().fillna(0)
        trades = df[df['posicao'] != 0].copy()

        if trades.empty or trades['posicao'].iloc[0] == -1: return self.retornar_metricas_vazias()
        # Remove a primeira operação se for uma saída
        if trades['posicao'].iloc[-1] == 1: trades = trades.iloc[:-1]
        # Garante que a última operação seja uma saída para fechar a posição
        entradas = trades[trades['posicao'] == 1]['preco']
        saidas = trades[trades['posicao'] == -1]['preco']

        # Alinha entradas e saídas
        if len(entradas) > len(saidas): entradas = entradas.iloc[:len(saidas)]
        # Calcula os retornos por operação, descontando os custos
        retornos = (saidas.values / entradas.values) - 1 - (self.custo_por_trade * 2)
        if len(retornos) == 0: return self.retornar_metricas_vazias()

        lucros = retornos[retornos > 0]
        perdas = retornos[retornos < 0]
        soma_lucros = np.sum(lucros)
        soma_perdas = np.abs(np.sum(perdas))
        profit_factor = soma_lucros / soma_perdas if soma_perdas > 0 else np.inf
        avg_win = np.mean(lucros) if len(lucros) > 0 else 0
        avg_loss = np.abs(np.mean(perdas)) if len(perdas) > 0 else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

        # Calcula as métricas de performance
        curva_equidade = np.cumprod(1 + retornos)
        capital_total = np.insert(curva_equidade, 0, 1)
        pico = np.maximum.accumulate(capital_total)
        drawdown_series = (capital_total - pico) / pico

        metricas = {
            'retorno_total': float(curva_equidade[-1] - 1),
            'trades': len(retornos),
            'sharpe': self.calculos.calcular_sharpe_ratio(retornos),
            'sortino': self.calculos.calcular_sortino_ratio(retornos),
            'max_drawdown': float(np.min(drawdown_series)),
            'win_rate': np.sum(retornos > 0) / len(retornos),
            'equity_curve': capital_total.tolist(),
            'retornos': retornos.tolist(),
            'profit_factor': float(profit_factor),
            'payoff_ratio': float(payoff_ratio),
            'drawdown_series': drawdown_series.tolist()
        }

        if verbose: logger.info(
            f"Backtest: {metricas['trades']} trades, Retorno: {metricas['retorno_total']:.2%}, Sharpe: {metricas['sharpe']:.2f}")
        return metricas