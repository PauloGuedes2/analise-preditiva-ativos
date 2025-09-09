import numpy as np


class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, df):
        """Cria features para o modelo - versão segura temporalmente"""
        df = df.copy()

        df['daily_return'] = df['Close'].pct_change()

        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()

        df['MA_ratio_5_20'] = df['MA_5'] / df['MA_20']
        df['MA_ratio_10_50'] = df['MA_10'] / df['MA_50']

        df['volatility_10'] = df['daily_return'].rolling(window=10, min_periods=1).std()
        df['volatility_20'] = df['daily_return'].rolling(window=20, min_periods=1).std()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        exp12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

        if 'Volume' in df.columns:
            df['volume_ma_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
            df['volume_ma_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']
            df['volume_change'] = df['Volume'].pct_change()

        for lag in [1, 2, 3, 5, 7]:
            df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

        df['target_class'] = (df['daily_return'].shift(-1) > 0).astype(int)
        df['target_price'] = df['Close'].shift(-1)
        df['target_return'] = df['daily_return'].shift(-1)

        df = df.dropna()

        return df

    def prepare_training_data(self, df):
        """Prepara dados para treinamento"""
        features = [
            'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_ratio_5_20', 'MA_ratio_10_50',
            'volatility_10', 'volatility_20', 'RSI', 'MACD', 'MACD_signal',
            'volume_ratio', 'volume_change', 'return_lag_1', 'return_lag_2', 'return_lag_3'
        ]

        available_features = [f for f in features if f in df.columns]

        X = df[available_features]
        y_class = df['target_class']
        y_price = df['target_price']
        y_return = df['target_return']

        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y_class = y_class.loc[X.index]
        y_price = y_price.loc[X.index]
        y_return = y_return.loc[X.index]

        return X, y_class, y_price, y_return, available_features
