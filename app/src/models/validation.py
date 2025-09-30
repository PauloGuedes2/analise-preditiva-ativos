from typing import Tuple, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedKFoldCV(KFold):
    """
    Validação Cruzada K-Fold com Purga para dados financeiros.

    Previne o vazamento de informações (data leakage) removendo amostras de treino
    cujos rótulos se sobrepõem no tempo com o período do conjunto de teste,
    uma causa comum de superestimação de performance em modelos de séries temporais.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series = None, purge_days: int = 1):
        """
        Inicializa o validador.

        Args:
            n_splits (int): Número de folds.
            t1 (pd.Series): Series contendo o timestamp de término de cada evento.
                            O índice deve corresponder ao de X.
            purge_days (int): Número de dias a serem removidos (purgados) entre
                              os conjuntos de treino e teste.
        """
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.purge_days = purge_days

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera os índices para dividir os dados em conjuntos de treino e teste.

        Args:
            X (pd.DataFrame): DataFrame de features.
            y (pd.Series, optional): Alvo. Não utilizado, presente por compatibilidade.
            groups (any, optional): Grupos. Não utilizado, presente por compatibilidade.

        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Uma tupla contendo os índices
                                                     de treino e teste para cada fold.
        """

        # Verificar se há dados suficientes para splits
        if len(X) < self.n_splits * 50:
            raise ValueError(f"Dados insuficientes para {self.n_splits} splits. "
                             f"Necessário mínimo de {self.n_splits * 50} amostras.")

        if not X.index.equals(self.t1.index):
            raise ValueError("O índice de X e o índice da série t1 devem ser idênticos para o PurgedKFoldCV funcionar.")

        indices = np.arange(X.shape[0])
        mbrg = self.purge_days

        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]
        for i, j in test_ranges:
            test_indices = indices[i:j]
            t1_test = self.t1.iloc[test_indices]

            test_start_time, test_end_time = X.index[i], t1_test.max()

            # 1. Purga: Remove do treino os eventos que terminam durante o período de teste.
            # Índices de treino são aqueles cujo evento termina ANTES do início do teste.
            train_indices_before = self.t1.index[self.t1 <= test_start_time].to_numpy()
            if train_indices_before.size > 0:
                last_train_idx_before = np.where(X.index == train_indices_before[-1])[0][0]
            else:
                last_train_idx_before = -1

            # 2. Embargo: Remove do treino os eventos que começam dentro do período de embargo após o teste.
            # Índices de treino são aqueles cujo evento começa APÓS o término do teste + embargo.
            train_indices_after = self.t1.index[self.t1 > test_end_time].to_numpy()
            if train_indices_after.size > 0:
                first_train_idx_after = np.where(X.index == train_indices_after[0])[0][0]
            else:
                first_train_idx_after = len(X)

            # Aplicar o embargo
            first_train_idx_after += mbrg

            # Combinar os índices de treino antes e depois do período de teste
            train_indices = np.concatenate(
                (indices[:last_train_idx_before + 1], indices[first_train_idx_after:])
            )

            # Garantir que os índices estejam dentro do intervalo válido
            train_indices = np.intersect1d(train_indices, np.arange(len(X)))

            yield train_indices, test_indices
