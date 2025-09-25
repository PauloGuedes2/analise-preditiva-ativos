from typing import Any

import numpy as np
import pandas as pd


class Utils:
    """Classe com utilitários gerais."""

    @staticmethod
    def converter_para_json_serializavel(obj: Any) -> Any:
        """Converte objetos para formato serializável em JSON."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return obj


class ValidadorDados:
    """Classe para validação de dados."""

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
