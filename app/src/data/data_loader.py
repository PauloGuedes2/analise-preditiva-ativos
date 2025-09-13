"""
Loader refinado de dados de mercado usando yfinance.
Baixa candles OHLCV direto do Yahoo Finance.
"""

import sqlite3

import pandas as pd
import yfinance as yf


class DataLoaderRefinado:
    def __init__(self, db_path: str = "dados_mercado.db"):
        self.db_path = db_path
        self._criar_tabelas()

    def _con(self):
        return sqlite3.connect(self.db_path)

    def _criar_tabelas(self):
        with self._con() as c:
            cur = c.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            c.commit()

    def baixar_dados_yf(self, ticker: str, periodo: str = "2y", intervalo: str = "1d") -> pd.DataFrame:
        """
        Baixa candles de um ticker via yfinance.
        :param ticker: código (ex: "PETR4.SA")
        :param periodo: janela (ex: "2y")
        :param intervalo: intervalo (ex: "1d")
        :return: DataFrame com OHLCV
        """
        df = yf.download(ticker, period=periodo, interval=intervalo, progress=False, auto_adjust=True)
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        # salvar no banco
        self.salvar_ohlcv(ticker, df)
        return df

    def salvar_ohlcv(self, ticker: str, df: pd.DataFrame):
        """Salva candles no banco SQLite"""
        with self._con() as c:
            cur = c.cursor()
            for idx, row in df.iterrows():
                open_val = float(row["Open"].iloc[0]) if hasattr(row["Open"], 'iloc') else float(row["Open"])
                high_val = float(row["High"].iloc[0]) if hasattr(row["High"], 'iloc') else float(row["High"])
                low_val = float(row["Low"].iloc[0]) if hasattr(row["Low"], 'iloc') else float(row["Low"])
                close_val = float(row["Close"].iloc[0]) if hasattr(row["Close"], 'iloc') else float(row["Close"])
                volume_val = float(row["Volume"].iloc[0]) if hasattr(row["Volume"], 'iloc') else float(row["Volume"])

                cur.execute("""
                    INSERT OR REPLACE INTO ohlcv (ticker,date,open,high,low,close,volume)
                    VALUES (?,?,?,?,?,?,?)
                """, (ticker, idx.strftime("%Y-%m-%d"), open_val, high_val, low_val, close_val, volume_val))
            c.commit()

    def carregar_do_bd(self, ticker: str) -> pd.DataFrame:
        """Carrega candles do banco"""
        with self._con() as c:
            df = pd.read_sql("SELECT * FROM ohlcv WHERE ticker=? ORDER BY date ASC", c, params=(ticker,))
        if len(df) == 0:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["Open", "High", "Low", "Close", "Volume"]]
