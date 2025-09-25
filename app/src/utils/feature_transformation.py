import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class FractionalDifferentiator:
    """
    Implementa a diferenciação fracionária para séries temporais.

    Este método visa tornar uma série estacionária (condição necessária para
    muitos modelos) enquanto preserva o máximo de "memória" (correlação)
    possível, ao contrário da diferenciação inteira que remove muita informação.
    """

    @staticmethod
    def get_weights_ffd(d: float, thres: float) -> np.ndarray:
        """
        Gera os pesos para a diferenciação fracionária de janela expansível (FFD).

        Args:
            d (float): A ordem da diferenciação (entre 0 e 1).
            thres (float): Limiar para truncar os pesos e otimizar o cálculo.

        Returns:
            np.ndarray: Um array de pesos a serem aplicados à série temporal.
        """
        w, k = [1.], 1
        while True:
            # Fórmula iterativa para calcular os pesos
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        # Inverte os pesos para a convolução
        return np.array(w[::-1]).reshape(-1, 1)

    @staticmethod
    def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
        """
        Aplica a diferenciação fracionária de janela expansível (FFD) a uma série.

        Args:
            series (pd.Series): A série temporal a ser diferenciada.
            d (float): A ordem da diferenciação.
            thres (float, optional): Limiar para os pesos. Defaults to 1e-5.

        Returns:
            pd.Series: A série diferenciada fracionariamente.
        """
        """Aplica a diferenciação fracionária de janela expansível."""
        w = FractionalDifferentiator.get_weights_ffd(d, thres)
        width = len(w) - 1

        df = pd.Series(dtype=float)
        # Itera sobre a série, aplicando os pesos em uma janela deslizante
        for iloc1 in range(width, series.shape[0]):
            loc0, loc1 = series.index[iloc1 - width], series.index[iloc1]
            if not np.isfinite(series.loc[loc1]):
                continue
            # Produto escalar entre os pesos e os valores da janela
            df[loc1] = np.dot(w.T, series.iloc[iloc1 - width:iloc1 + 1])[0]
        return df

    @staticmethod
    def find_min_d(series: pd.Series, max_d: float = 1.0, step: float = 0.01) -> float:
        """
        Encontra a menor ordem de diferenciação 'd' que torna a série estacionária.

        A estacionariedade é verificada usando o teste Augmented Dickey-Fuller (ADF).

        Args:
            series (pd.Series): A série temporal.
            max_d (float, optional): A ordem máxima de 'd' a ser testada.
            step (float, optional): O incremento para testar 'd'.

        Returns:
            float: A menor ordem 'd' que resulta em estacionariedade.
        """
        for d in np.arange(0, max_d + step, step):
            d_series = FractionalDifferentiator.frac_diff_ffd(series.to_frame('Close')['Close'], d)

            d_series_cleaned = d_series.dropna()

            if len(d_series_cleaned) < 20:
                continue # Pula se houver poucos dados para o teste ADF

            # Realiza o teste ADF na série diferenciada
            adf_result = adfuller(d_series_cleaned, maxlag=1, regression='c', autolag=None)

            # Se o p-valor for menor que 5%, a série é considerada estacionária
            if adf_result[1] < 0.05:  # p-value < 5%
                return d
        return max_d  # Retorna max_d se nenhuma ordem ótima for encontrada
