import pandas as pd


class Utils:
    """Classe utilitária com métodos estáticos diversos."""

    @staticmethod
    def validar_dados_treinamento(X: pd.DataFrame, y: pd.Series,
                                  min_amostras: int = 100) -> bool:
        """Valida dados para treinamento."""
        if len(X) < min_amostras or len(y) < min_amostras:
            return False

        if len(X) != len(y):
            return False

        if y.nunique() < 2:  # Pelo menos duas classes
            return False

        return True
