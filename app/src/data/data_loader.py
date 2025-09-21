import os
import sqlite3
from contextlib import contextmanager
from typing import Tuple

import pandas as pd
import yfinance as yf

from src.config.params import Params
from src.logger.logger import logger


class DataLoader:
    """Carrega e gerencia dados de mercado do Yahoo Finance."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or Params.PATH_DB_MERCADO
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._criar_tabelas()

    @contextmanager
    def _conexao(self):
        """Context manager para gerenciar conexões com o banco."""
        conexao = sqlite3.connect(self.db_path)
        try:
            yield conexao
        finally:
            conexao.close()

    def _criar_tabelas(self):
        """Cria tabelas necessárias se não existirem."""
        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute("""
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
            conn.commit()

    @staticmethod
    def _processar_dados_yfinance(dados_completos: pd.DataFrame, ticker: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """Processa dados brutos do yfinance e separa em DataFrames."""
        df_ticker = pd.DataFrame({
            'Open': dados_completos['Open'][ticker],
            'High': dados_completos['High'][ticker],
            'Low': dados_completos['Low'][ticker],
            'Close': dados_completos['Close'][ticker],
            'Volume': dados_completos['Volume'][ticker]
        }).dropna()

        df_ibov = dados_completos['Close']['^BVSP'].to_frame('Close_IBOV')

        return df_ticker, df_ibov

    def baixar_dados_yf(self, ticker: str, periodo: str = None,
                        intervalo: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Baixa dados do ticker e do IBOVESPA e os salva no BD."""
        periodo = periodo or Params.PERIODO_DADOS
        intervalo = intervalo or Params.INTERVALO_DADOS

        logger.info(f"Baixando dados para {ticker} - Período: {periodo}, Intervalo: {intervalo}")

        try:
            dados_completos = yf.download(
                f"{ticker} ^BVSP",
                period=periodo,
                interval=intervalo,
                progress=False,
                auto_adjust=True,
                timeout=30
            )

            if dados_completos.empty or ticker not in dados_completos['Close'].columns:
                raise ValueError(f"Nenhum dado retornado para o ticker {ticker}.")

            df_ticker, df_ibov = self._processar_dados_yfinance(dados_completos, ticker)
            self.salvar_ohlcv(ticker, df_ticker)

            logger.info(f"Dados baixados - {ticker}: {len(df_ticker)} registros")
            return df_ticker, df_ibov

        except Exception as e:
            logger.error(f"Erro crítico ao baixar dados do yfinance para {ticker}: {e}")
            raise

        except Exception as e:
            logger.error(f"Erro crítico ao baixar dados do yfinance: {e}")
            # Tentar carregar do banco de dados se disponível
            df_bd = self.carregar_do_bd(ticker)
            if not df_bd.empty:
                logger.info(f"Usando dados do BD para {ticker}: {len(df_bd)} registros")
                return df_bd, pd.DataFrame()
            raise Exception(f"Erro ao baixar dados e nenhum backup disponível: {e}")

    def salvar_ohlcv(self, ticker: str, df: pd.DataFrame):
        """Salva dados OHLCV no banco de dados."""
        with self._conexao() as conn:
            cursor = conn.cursor()

            for data, linha in df.iterrows():
                valores = (
                    ticker,
                    data.strftime("%Y-%m-%d"),
                    float(linha["Open"]),
                    float(linha["High"]),
                    float(linha["Low"]),
                    float(linha["Close"]),
                    float(linha["Volume"])
                )

                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, valores)

            conn.commit()

        logger.info(f"Dados salvos no BD - {ticker}: {len(df)} registros")

    def carregar_do_bd(self, ticker: str) -> pd.DataFrame:
        """Carrega dados OHLCV do banco de dados."""
        with self._conexao() as conn:
            query = "SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date ASC"
            df = pd.read_sql(query, conn, params=(ticker,))

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Dados carregados do BD - {ticker}: {len(df)} registros")
        return df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]

    def verificar_dados_disponiveis(self, ticker: str) -> bool:
        """Verifica se existem dados para um ticker específico."""
        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE ticker = ?",
                (ticker,)
            )
            count = cursor.fetchone()[0] > 0

        logger.info(f"Verificação de dados - {ticker}: {'Disponível' if count else 'Indisponível'}")
        return count
