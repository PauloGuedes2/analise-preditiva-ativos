import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import yfinance as yf

from src.config.params import Params
from src.logger.logger import logger


class DataLoader:
    """Carrega e gerencia dados de mercado do Yahoo Finance, com cache em um banco de dados local."""

    def __init__(self, db_path: str = None):
        """
        Inicializa o DataLoader.

        Args:
            db_path (str, optional): Caminho para o arquivo do banco de dados de mercado.
                                     Se não fornecido, usa o padrão de `Params`.
        """
        self.db_path = db_path or Params.PATH_DB_MERCADO
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._criar_tabelas()

    @contextmanager
    def _conexao(self):
        """Context manager para gerenciar conexões com o banco de dados SQLite."""
        conexao = sqlite3.connect(self.db_path)
        try:
            yield conexao
        finally:
            conexao.close()

    def _criar_tabelas(self):
        """Cria a tabela `ohlcv` para armazenar os dados de mercado, se não existir."""
        with self._conexao() as conn:
            cursor = conn.cursor()
            # Chave primária composta por ticker e data para evitar duplicatas
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
        """
        Processa o DataFrame bruto do yfinance, separando os dados do ticker e do IBOV.

        Args:
            dados_completos (pd.DataFrame): DataFrame com MultiIndex de yfinance.
            ticker (str): O nome do ticker principal.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Um DataFrame para o ticker e outro para o IBOV.
        """
        # Extrai colunas específicas do ticker
        df_ticker = pd.DataFrame({
            'Open': dados_completos['Open'][ticker],
            'High': dados_completos['High'][ticker],
            'Low': dados_completos['Low'][ticker],
            'Close': dados_completos['Close'][ticker],
            'Volume': dados_completos['Volume'][ticker]
        }).dropna()

        # Extrai a coluna de fechamento do IBOV
        df_ibov = dados_completos['Close']['^BVSP'].to_frame('Close_IBOV')

        return df_ticker, df_ibov

    def baixar_dados_yf(self, ticker: str, periodo: str = None,
                        intervalo: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Baixa dados do yfinance de forma robusta, especificando datas de início
        e fim para evitar erros de formatação de data relacionados à localidade do sistema.
        """
        intervalo = intervalo or Params.INTERVALO_DADOS
        periodo_config = periodo or Params.PERIODO_DADOS

        end_date = datetime.now() + timedelta(days=1)

        # Converte o período (ex: "3y") em uma data de início
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

    def atualizar_dados_ticker(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Atualiza os dados do ticker e do IBOV, salvando no banco de dados.
        Retorna os dados atualizados. Levanta exceção em caso de falha.
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

            # Salva no banco de dados
            self.salvar_ohlcv(ticker, df_ticker)

            logger.info(f"Dados atualizados com sucesso para {ticker}")
            return df_ticker, df_ibov

        except Exception as e:
            logger.error(f"Erro ao atualizar dados para {ticker}: {e}")
            raise
