from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedKFoldCV(KFold):
    """
    Validação Cruzada K-Fold com Purga e Embargo para dados financeiros.
    Previne o vazamento de dados removendo amostras de treino cujos rótulos
    se sobrepõem no tempo com o período do conjunto de teste.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series = None, purge_days: int = 1):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.purge_days = purge_days

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        # Verificar se há dados suficientes para splits
        if len(X) < self.n_splits * 50:  # Mínimo 50 amostras por split
            raise ValueError(f"Dados insuficientes para {self.n_splits} splits. "
                             f"Necessário mínimo de {self.n_splits * 50} amostras.")

        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("Índices de X e t1 devem ser idênticos.")

        indices = np.arange(X.shape[0])
        mbrg = self.purge_days

        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]
        for i, j in test_ranges:
            test_indices = indices[i:j]
            t1_test = self.t1.iloc[test_indices]

            # Obter timestamps de início e fim do conjunto de teste
            test_start_time = self.t1.index[i]
            test_end_time = t1_test.max()

            # 1. Purga no início do conjunto de treino (antes do teste)
            train_indices_before = self.t1.index[self.t1 <= test_start_time].to_numpy()
            if train_indices_before.size > 0:
                last_train_idx_before = np.where(X.index == train_indices_before[-1])[0][0]
            else:
                last_train_idx_before = -1

            # 2. Purga no final do conjunto de treino (depois do teste)
            train_indices_after = self.t1.index[self.t1 > test_end_time].to_numpy()
            if train_indices_after.size > 0:
                first_train_idx_after = np.where(X.index == train_indices_after[0])[0][0]
            else:
                first_train_idx_after = len(X)

            # Aplicar embargo
            first_train_idx_after += mbrg

            # Combinar os índices de treino antes e depois do teste (com purga e embargo)
            train_indices = np.concatenate(
                (indices[:last_train_idx_before + 1], indices[first_train_idx_after:])
            )

            # Filtrar para garantir que os índices de treino existam em X
            train_indices = np.intersect1d(train_indices, np.arange(len(X)))

            yield train_indices, test_indices
