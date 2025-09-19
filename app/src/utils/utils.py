import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, List

import numpy as np
import pandas as pd


class Utils:
    """Classe com utilitários gerais."""

    @staticmethod
    def criar_diretorio(caminho: str) -> bool:
        """Cria diretório se não existir."""
        try:
            os.makedirs(caminho, exist_ok=True)
            return True
        except Exception as e:
            print(f"Erro ao criar diretório {caminho}: {e}")
            return False

    @staticmethod
    def validar_dataframe(df: pd.DataFrame, colunas_necessarias: List[str]) -> bool:
        """Valida se DataFrame possui colunas necessárias."""
        if df.empty:
            return False

        colunas_existentes = set(df.columns)
        colunas_requeridas = set(colunas_necessarias)

        return colunas_requeridas.issubset(colunas_existentes)

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

    @staticmethod
    def formatar_percentual(valor: float, casas_decimais: int = 2) -> str:
        """Formata número como percentual."""
        return f"{valor:.{casas_decimais}%}"

    @staticmethod
    def formatar_moeda(valor: float, simbolo: str = "R$") -> str:
        """Formata número como valor monetário."""
        return f"{simbolo} {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

    @staticmethod
    def calcular_tempo_execucao(inicio: datetime, fim: datetime) -> str:
        """Calcula e formata tempo de execução."""
        duracao = fim - inicio
        segundos = duracao.total_seconds()

        if segundos < 60:
            return f"{segundos:.1f} segundos"
        elif segundos < 3600:
            return f"{segundos / 60:.1f} minutos"
        else:
            return f"{segundos / 3600:.1f} horas"


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

    @staticmethod
    def detectar_outliers(serie: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detecta outliers usando método Z-score."""
        z_scores = np.abs((serie - serie.mean()) / serie.std())
        return z_scores > threshold


@contextmanager
def gerenciar_recursos(*recursos):
    """
    Context manager para gerenciar múltiplos recursos.

    Exemplo:
    with gerenciar_recursos(conn1, conn2, file) as (c1, c2, f):
        # usar recursos
    """
    try:
        yield recursos
    finally:
        for recurso in recursos:
            if hasattr(recurso, 'close'):
                recurso.close()


def timer(func):
    """Decorator para medir tempo de execução de funções."""
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fim = time.time()
        print(f"⏱️  {func.__name__} executado em {fim - inicio:.2f} segundos")
        return resultado

    return wrapper
