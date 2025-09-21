import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class FractionalDifferentiator:
    """
    Aplica diferenciação fracionária a uma série temporal para torná-la estacionária
    enquanto preserva a memória.
    """

    @staticmethod
    def get_weights_ffd(d: float, thres: float) -> np.ndarray:
        """Gera os pesos para a diferenciação fracionária."""
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    @staticmethod
    def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
        """Aplica a diferenciação fracionária de janela expansível."""
        w = FractionalDifferentiator.get_weights_ffd(d, thres)
        width = len(w) - 1

        df = pd.Series(dtype=float)
        for iloc1 in range(width, series.shape[0]):
            loc0, loc1 = series.index[iloc1 - width], series.index[iloc1]
            if not np.isfinite(series.loc[loc1]):
                continue
            df[loc1] = np.dot(w.T, series.iloc[iloc1 - width:iloc1 + 1])[0]
        return df

    @staticmethod
    def find_min_d(series: pd.Series, max_d: float = 1.0, step: float = 0.01) -> float:
        """
        Encontra a menor ordem de diferenciação 'd' que torna a série estacionária.
        """
        for d in np.arange(0, max_d + step, step):
            d_series = FractionalDifferentiator.frac_diff_ffd(series.to_frame('Close')['Close'], d)

            d_series_cleaned = d_series.dropna()

            if len(d_series_cleaned) < 20:
                continue

            adf_result = adfuller(d_series_cleaned, maxlag=1, regression='c', autolag=None)
            if adf_result[1] < 0.05:  # p-value < 5%
                return d
        return max_d  # Retorna max_d se nenhuma ordem ótima for encontrada
