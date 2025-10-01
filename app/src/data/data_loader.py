import os
import re
from urllib.parse import urlparse

import psycopg2
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import yfinance as yf

from src.config.params import Params
from src.logger.logger import logger


class DataLoader:
    """Carrega e gerencia dados de mercado, com cache em PostgreSQL."""

    def __init__(self, db_config: dict = None):
        self.db_config = db_config
        self._criar_tabelas()

    @contextmanager
    def _conexao(self):
        """Context manager para gerenciar conexões com o PostgreSQL."""
        url = urlparse("postgresql://admin:Le0t7F5gLxYIPS1gSsnuBum1oR42CYbb@dpg-d3e54s2li9vc739bs410-a.oregon-postgres.render.com/acoes")
        conn = psycopg2.connect(
            dbname=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port,
            sslmode="require"
        )
        try:
            yield conn
        finally:
            conn.close()

    def _criar_tabelas(self):
        """Cria a tabela `ohlcv` para armazenar os dados de mercado, se não existir."""
        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.commit()

    @staticmethod
    def _processar_dados_yfinance(dados_completos: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processa o DataFrame bruto do yfinance, separando os dados do ticker e do IBOV.
        """
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
        """
        Baixa dados do yfinance e salva no banco.
        """
        intervalo = intervalo or Params.INTERVALO_DADOS
        periodo_config = periodo or Params.PERIODO_DADOS

        end_date = datetime.now() + timedelta(days=1)

        # Converte período ("3y", "6mo", "10d") em data inicial
        match = re.match(r"(\d+)(\w+)", periodo_config)
        if not match:
            raise ValueError(f"Formato de período inválido: '{periodo_config}'. Use '4y', '6mo', '10d', etc.")

        valor, unidade = int(match.group(1)), match.group(2).lower()
        if unidade == 'y':
            start_date = end_date - timedelta(days=valor * 365)
        elif unidade in ['mo', 'm']:
            start_date = end_date - timedelta(days=valor * 30)
        elif unidade == 'd':
            start_date = end_date - timedelta(days=valor)
        else:
            raise ValueError(f"Unidade de período não suportada: '{unidade}'. Use 'y', 'mo' ou 'd'.")

        start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        logger.info(f"Baixando dados para {ticker} - De: {start_str} até {end_str}")

        try:
            dados_completos = yf.download(
                tickers=f"{ticker} ^BVSP",
                start=start_str,
                end=end_str,
                interval=intervalo,
                progress=False,
                auto_adjust=True,
                timeout=30
            )

            if dados_completos.empty or 'Close' not in dados_completos.columns or ticker not in dados_completos[
                'Close'].columns:
                raise ValueError(f"Nenhum dado retornado para o ticker {ticker}.")

            df_ticker, df_ibov = self._processar_dados_yfinance(dados_completos, ticker)
            self.salvar_ohlcv(ticker, df_ticker)

            logger.info(f"Dados baixados - {ticker}: {len(df_ticker)} registros")
            return df_ticker, df_ibov

        except Exception as e:
            logger.error(f"Erro crítico ao baixar dados do yfinance para {ticker}: {e}")
            raise

    def salvar_ohlcv(self, ticker: str, df: pd.DataFrame):
        """Salva dados OHLCV no banco PostgreSQL."""
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
                    INSERT INTO ohlcv (ticker, date, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, valores)

            conn.commit()

        logger.info(f"Dados salvos no BD - {ticker}: {len(df)} registros")

    def carregar_do_bd(self, ticker: str) -> pd.DataFrame:
        """Carrega dados OHLCV do banco PostgreSQL."""
        with self._conexao() as conn:
            query = "SELECT * FROM ohlcv WHERE ticker = %s ORDER BY date ASC"
            df = pd.read_sql(query, conn, params=(ticker,))

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Dados carregados do BD - {ticker}: {len(df)} registros")
        return df.set_index("date")[["open", "high", "low", "close", "volume"]]

    def verificar_dados_disponiveis(self, ticker: str) -> bool:
        """Verifica se existem dados para um ticker específico."""
        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE ticker = %s", (ticker,))
            count = cursor.fetchone()[0] > 0

        logger.info(f"Verificação de dados - {ticker}: {'Disponível' if count else 'Indisponível'}")
        return count

    def atualizar_dados_ticker(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Atualiza os dados do ticker e do IBOV, salvando no banco de dados. Retorna os dados atualizados.
        """
        logger.info(f"Atualizando dados para {ticker}...")
        try:
            dados_completos = yf.download(
                f"{ticker} ^BVSP",
                period=Params.PERIODO_DADOS,
                interval=Params.INTERVALO_DADOS,
                progress=False,
                auto_adjust=True,
                timeout=15
            )
            if dados_completos.empty:
                raise ValueError(f"Nenhum dado novo retornado para {ticker} do yfinance.")

            df_ticker, df_ibov = self._processar_dados_yfinance(dados_completos, ticker)
            self.salvar_ohlcv(ticker, df_ticker)
            logger.info(f"Dados atualizados com sucesso para {ticker}")
            return df_ticker, df_ibov

        except Exception as e:
            logger.error(f"Erro ao atualizar dados para {ticker}: {e}")
            raise e