import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, df):
        """Cria features para o modelo"""
        df = df.copy()

        # Retorno diário
        df['daily_return'] = df['close'].pct_change()

        # Alvo: 1 se próximo dia > 0, senão 0
        df['target'] = (df['daily_return'].shift(-1) > 0).astype(int)

        # Features adicionais
        df['return_lag1'] = df['daily_return'].shift(1)
        df['return_lag2'] = df['daily_return'].shift(2)
        df['return_lag3'] = df['daily_return'].shift(3)

        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_lag1'] = df['volume_change'].shift(1)

        # Remove linhas com NaN
        df = df.dropna()

        return df

    def prepare_training_data(self, df):
        """Prepara dados para treinamento"""
        features = ['return_lag1', 'return_lag2', 'return_lag3', 'volume_lag1']

        # Verificar se todas as features existem
        available_features = [f for f in features if f in df.columns]

        x = df[available_features]
        y = df['target']

        return x, y, available_features