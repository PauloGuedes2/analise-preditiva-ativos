# risk_analyzer_refinado.py
"""
Backtest simples e métricas de risco
- backtest_sinais: recebe df com 'preco','pred' (1 para comprar/short not supported here),
  aplica lógica simples: ao sinal 1 -> entra long no fechamento e fecha no próximo dia (intraday simulado).
- retorna métricas: retorno total, retorno anualizado aproximado, sharpe (uso desvio simples), max drawdown
"""

import numpy as np
import pandas as pd


class RiskAnalyzerRefinado:
    def __init__(self):
        pass

    def backtest_sinais(self, df_signals: pd.DataFrame, custo_por_trade_pct: float = 0.0005) -> dict:
        """
        df_signals: DataFrame com colunas ['preco','proba','pred'] index alinhado ao tempo
        Estratégia: quando pred==1 -> entra long no preço de fechamento atual e fecha no próximo índice
        Simples, instrutivo — adaptar para regras reais (stop, sl, target, posição size, etc.)
        """
        if len(df_signals) < 2:
            return {'retorno_total': 0.0, 'trades': 0, 'sharpe': 0.0,
                    'max_drawdown': 0.0, 'retorno_med_diario': 0.0}

        df = df_signals.reset_index(drop=True).copy()
        n = len(df)
        rets = []
        trades = 0
        for i in range(n - 1):
            if df.loc[i, 'pred'] == 1:
                preco_enter = df.loc[i, 'preco']
                preco_exit = df.loc[i + 1, 'preco']  # close next day
                ret = (preco_exit / preco_enter) - 1.0
                # aplicar custo (enter + exit)
                ret -= 2 * custo_por_trade_pct
                rets.append(ret)
                trades += 1
        if len(rets) == 0:
            return {'retorno_total': 0.0, 'trades': 0, 'sharpe': 0.0, 'max_drawdown': 0.0, 'retorno_med_diario': 0.0}
        arr = np.array(rets)
        retorno_total = np.prod(1 + arr) - 1
        retorno_med_diario = np.mean(arr)
        sharpe = (np.mean(arr) / (np.std(arr) + 1e-9)) * np.sqrt(252) if np.std(arr) > 0 else 0.0
        # equity curve
        equity = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        max_draw = float(np.min(drawdowns))
        return {
            'retorno_total': float(retorno_total),
            'trades': int(trades),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_draw),
            'retorno_med_diario': float(retorno_med_diario)
        }
