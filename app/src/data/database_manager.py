import os
import sqlite3
from datetime import datetime

import pandas as pd


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
            date TEXT,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at TEXT,
            PRIMARY KEY (date, ticker)
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
        """Salva dados da ação no banco - versão simplificada"""
        if df.empty:
            print("DataFrame vazio. Nada para salvar.")
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        created_at = datetime.now().isoformat()
        records = []

        for index, row in df.iterrows():
            try:
                if isinstance(index, tuple):
                    date_str = index[0].strftime('%Y-%m-%d') if hasattr(index[0], 'strftime') else str(index[0])
                else:
                    date_str = index.strftime('%Y-%m-%d') if hasattr(index, 'strftime') else str(index)

                records.append((
                    date_str,
                    ticker,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']),
                    created_at
                ))
            except Exception as e:
                print(f"Erro ao processar linha {index}: {e}")
                continue

        try:
            if records:
                cursor.executemany('''
                INSERT OR REPLACE INTO stock_prices 
                (date, ticker, open, high, low, close, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', records)

                print(f"Salvos {len(records)} registros para {ticker} no banco de dados")
                conn.commit()
            else:
                print("Nenhum registro válido para salvar")

        except Exception as e:
            print(f"Erro ao salvar dados: {e}")
            conn.rollback()

        conn.close()

    def load_stock_data(self, ticker):
        """Carrega dados da ação do banco"""
        conn = self._get_connection()

        try:
            query = "SELECT date, open, high, low, close, volume FROM stock_prices WHERE ticker = ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=[ticker], parse_dates=['date'])

            if not df.empty:
                df.set_index('date', inplace=True)
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)

                print(f"Carregados {len(df)} registros para {ticker} do banco de dados")
            else:
                print(f"Nenhum dado encontrado para {ticker} no banco de dados")

        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            df = pd.DataFrame()

        finally:
            conn.close()

        return df

    def save_prediction(self, ticker, prediction_date, prediction, confidence):
        """Salva uma previsão no banco de dados"""
        conn = self._get_connection()
        cursor = conn.cursor()

        created_at = datetime.now().isoformat()

        try:
            cursor.execute('''
            INSERT INTO predictions (ticker, prediction_date, prediction, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (ticker, prediction_date, prediction, confidence, created_at))

            conn.commit()
            print(f"Previsão salva para {ticker} na data {prediction_date}")

        except Exception as e:
            print(f"Erro ao salvar previsão: {e}")
            conn.rollback()

        finally:
            conn.close()

    def get_predictions(self, ticker, limit=10):
        """Recupera as últimas previsões para um ticker"""
        conn = self._get_connection()

        try:
            query = """
            SELECT prediction_date, prediction, confidence, created_at 
            FROM predictions 
            WHERE ticker = ? 
            ORDER BY created_at DESC 
            LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=[ticker, limit], parse_dates=['prediction_date', 'created_at'])
            return df

        except Exception as e:
            print(f"Erro ao carregar previsões: {e}")
            return pd.DataFrame()

        finally:
            conn.close()
