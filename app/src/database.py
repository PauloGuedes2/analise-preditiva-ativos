import sqlite3
import pandas as pd
from datetime import datetime
import os


class DatabaseManager:
    def __init__(self, db_path='src/data/database.db'):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_database()

    def _get_connection(self):
        """Retorna uma conexão com o banco de dados"""
        return sqlite3.connect(self.db_path)

    def _init_database(self):
        """Inicializa o banco de dados e cria tabelas se não existirem"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            date TEXT PRIMARY KEY,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at TEXT
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            prediction_date TEXT,
            prediction INTEGER,
            confidence REAL,
            created_at TEXT
        )
        ''')

        conn.commit()
        conn.close()

    def save_stock_data(self, df, ticker):
        """Salva dados da ação no banco"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Preparar dados para inserção
        created_at = datetime.now().isoformat()

        for index, row in df.iterrows():
            try:
                cursor.execute('''
                INSERT OR REPLACE INTO stock_prices 
                (date, ticker, open, high, low, close, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    index.strftime('%Y-%m-%d'),
                    ticker,
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    created_at
                ))
            except Exception as e:
                print(f"Erro ao inserir dados para {index}: {e}")

        conn.commit()
        conn.close()

    def load_stock_data(self, ticker):
        """Carrega dados da ação do banco"""
        conn = self._get_connection()
        query = f"SELECT date, open, high, low, close, volume FROM stock_prices WHERE ticker = '{ticker}' ORDER BY date"

        try:
            df = pd.read_sql_query(query, conn, parse_dates=['date'])
            if not df.empty:
                df.set_index('date', inplace=True)
                # Renomear colunas para formato padrão
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            df = pd.DataFrame()

        conn.close()
        return df