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

    def baixar_dados_yf(self, ticker: str, periodo: str = "3y", intervalo: str = "1d") -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Baixa candles de um ticker e do IBOVESPA via yfinance.
        :return: tupla com (DataFrame do Ticker, DataFrame do IBOV)
        """
        # Baixar dados do ticker e do Ibovespa em um único request
        df_completo = yf.download(f"{ticker} ^BVSP", period=periodo, interval=intervalo, progress=False,
                                  auto_adjust=True)

        # Separar os dataframes
        df_ticker = df_completo['Close'][ticker].to_frame('Close')
        df_ticker['Open'] = df_completo['Open'][ticker]
        df_ticker['High'] = df_completo['High'][ticker]
        df_ticker['Low'] = df_completo['Low'][ticker]
        df_ticker['Volume'] = df_completo['Volume'][ticker]

        df_ibov = df_completo['Close']['^BVSP'].to_frame('Close_IBOV')

        # Limpar e alinhar dados do ticker
        df_ticker = df_ticker[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df_ticker.index = pd.to_datetime(df_ticker.index)

        # Salvar no banco (apenas o ticker, como antes)
        self.salvar_ohlcv(ticker, df_ticker)

        # Retornar ambos os dataframes
        return df_ticker, df_ibov

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
