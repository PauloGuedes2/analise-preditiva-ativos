import os
import sqlite3
from datetime import datetime

import pandas as pd


class GerenciadorBancoDados:
    """
    Classe responsável por gerenciar o banco de dados SQLite para armazenamento de dados de ações e previsões.
    
    Esta classe fornece métodos para criar, salvar e recuperar dados históricos de ações,
    bem como armazenar e consultar previsões do modelo de machine learning.
    """
    
    def __init__(self, caminho_bd='src/data/database.db'):
        """
        Inicializa o gerenciador de banco de dados.
        
        Args:
            caminho_bd (str): Caminho para o arquivo do banco de dados SQLite
        """
        os.makedirs(os.path.dirname(caminho_bd), exist_ok=True)
        self.caminho_bd = caminho_bd
        self._inicializar_banco()

    def _obter_conexao(self):
        """
        Retorna uma conexão com o banco de dados SQLite.
        
        Returns:
            sqlite3.Connection: Conexão ativa com o banco de dados
        """
        return sqlite3.connect(self.caminho_bd)

    def _inicializar_banco(self):
        """
        Inicializa o banco de dados e cria as tabelas necessárias se não existirem.
        
        Cria duas tabelas principais:
        - precos_acoes: Para armazenar dados históricos de preços
        - previsoes: Para armazenar previsões do modelo
        """
        conn = self._obter_conexao()
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS precos_acoes (
            data TEXT,
            ticker TEXT,
            abertura REAL,
            maxima REAL,
            minima REAL,
            fechamento REAL,
            volume INTEGER,
            criado_em TEXT,
            PRIMARY KEY (data, ticker)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            data_previsao TEXT,
            previsao INTEGER,
            confianca REAL,
            criado_em TEXT
        )
        ''')

        conn.commit()
        conn.close()

    def salvar_dados_acao(self, df, ticker):
        """
        Salva dados históricos de uma ação no banco de dados.
        
        Args:
            df (pandas.DataFrame): DataFrame com dados históricos (Open, High, Low, Close, Volume)
            ticker (str): Símbolo da ação (ex: 'PETR4.SA')
        """
        if df.empty:
            print("DataFrame vazio. Nada para salvar.")
            return

        conn = self._obter_conexao()
        cursor = conn.cursor()

        criado_em = datetime.now().isoformat()
        registros = []

        for indice, linha in df.iterrows():
            try:
                if isinstance(indice, tuple):
                    data_str = indice[0].strftime('%Y-%m-%d') if hasattr(indice[0], 'strftime') else str(indice[0])
                else:
                    data_str = indice.strftime('%Y-%m-%d') if hasattr(indice, 'strftime') else str(indice)

                registros.append((
                    data_str,
                    ticker,
                    float(linha['Open']),
                    float(linha['High']),
                    float(linha['Low']),
                    float(linha['Close']),
                    int(linha['Volume']),
                    criado_em
                ))
            except Exception as e:
                print(f"Erro ao processar linha {indice}: {e}")
                continue

        try:
            if registros:
                cursor.executemany('''
                INSERT OR REPLACE INTO precos_acoes 
                (data, ticker, abertura, maxima, minima, fechamento, volume, criado_em)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', registros)

                print(f"Salvos {len(registros)} registros para {ticker} no banco de dados")
                conn.commit()
            else:
                print("Nenhum registro válido para salvar")

        except Exception as e:
            print(f"Erro ao salvar dados: {e}")
            conn.rollback()

        conn.close()

    def carregar_dados_acao(self, ticker):
        """
        Carrega dados históricos de uma ação do banco de dados.
        
        Args:
            ticker (str): Símbolo da ação (ex: 'PETR4.SA')
            
        Returns:
            pandas.DataFrame: DataFrame com dados históricos da ação
        """
        conn = self._obter_conexao()

        try:
            consulta = "SELECT data, abertura, maxima, minima, fechamento, volume FROM precos_acoes WHERE ticker = ? ORDER BY data"
            df = pd.read_sql_query(consulta, conn, params=[ticker], parse_dates=['data'])

            if not df.empty:
                df.set_index('data', inplace=True)
                df.rename(columns={
                    'abertura': 'Open',
                    'maxima': 'High',
                    'minima': 'Low',
                    'fechamento': 'Close',
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

    def salvar_previsao(self, ticker, data_previsao, previsao, confianca):
        """
        Salva uma previsão do modelo no banco de dados.
        
        Args:
            ticker (str): Símbolo da ação (ex: 'PETR4.SA')
            data_previsao (str): Data da previsão no formato 'YYYY-MM-DD'
            previsao (int): Previsão (0 para baixa, 1 para alta)
            confianca (float): Nível de confiança da previsão (0.0 a 1.0)
        """
        conn = self._obter_conexao()
        cursor = conn.cursor()

        criado_em = datetime.now().isoformat()

        try:
            cursor.execute('''
            INSERT INTO previsoes (ticker, data_previsao, previsao, confianca, criado_em)
            VALUES (?, ?, ?, ?, ?)
            ''', (ticker, data_previsao, previsao, confianca, criado_em))

            conn.commit()
            print(f"Previsão salva para {ticker} na data {data_previsao}")

        except Exception as e:
            print(f"Erro ao salvar previsão: {e}")
            conn.rollback()

        finally:
            conn.close()

    def obter_previsoes(self, ticker, limite=10):
        """
        Recupera as últimas previsões para um ticker específico.
        
        Args:
            ticker (str): Símbolo da ação (ex: 'PETR4.SA')
            limite (int): Número máximo de previsões a retornar
            
        Returns:
            pandas.DataFrame: DataFrame com as previsões mais recentes
        """
        conn = self._obter_conexao()

        try:
            consulta = """
            SELECT data_previsao, previsao, confianca, criado_em 
            FROM previsoes 
            WHERE ticker = ? 
            ORDER BY criado_em DESC 
            LIMIT ?
            """
            df = pd.read_sql_query(consulta, conn, params=[ticker, limite], parse_dates=['data_previsao', 'criado_em'])
            return df

        except Exception as e:
            print(f"Erro ao carregar previsões: {e}")
            return pd.DataFrame()

        finally:
            conn.close()
