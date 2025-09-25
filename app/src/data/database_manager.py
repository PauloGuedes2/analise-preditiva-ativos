import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any

from src.config.params import Params
from src.logger.logger import logger
from src.utils.utils import Utils


class DatabaseManager:
    """Gerencia operações de banco de dados SQLite para metadados do modelo e previsões."""

    def __init__(self, db_path: str = None):
        """
        Inicializa o gerenciador de banco de dados.

        Args:
            db_path (str, optional): Caminho para o arquivo do banco de dados.
                                     Se não fornecido, usa o padrão de `Params`.
        """
        self.db_path = db_path or Params.PATH_DB_METADATA
        # Garante que o diretório para o DB exista
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._criar_tabelas()
        logger.info(f"DatabaseManager inicializado com banco: {self.db_path}")

    @contextmanager
    def _conexao(self):
        """Context manager para gerenciar conexões com o banco."""
        conexao = sqlite3.connect(self.db_path)
        try:
            yield conexao
        finally:
            conexao.close()

    def _criar_tabelas(self):
        """Cria a tabela `previsoes` se ela não existir."""
        try:
            with self._conexao() as conn:
                cursor = conn.cursor()

                # Tabela para armazenar cada previsão gerada
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS previsoes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        criado_em TEXT,
                        predicao INTEGER,
                        probabilidade REAL,
                        metadados_json TEXT
                    )
                """)

                conn.commit()
            logger.debug("Tabelas de metadados criadas/verificadas com sucesso")

        except Exception as e:
            logger.error(f"Erro ao criar tabelas: {e}")
            raise

    @staticmethod
    def _obter_timestamp_atual() -> str:
        """Retorna o timestamp UTC atual em formato de string ISO."""
        return datetime.utcnow().isoformat()

    def salvar_previsao(self, dados: Dict[str, Any]):
        """Salva os dados de uma única previsão no banco de dados."""
        try:
            timestamp = self._obter_timestamp_atual()
            predicao = dados.get('predicao')
            probabilidade = dados.get('probabilidade')
            metadados = dados.get('metadados', {})
            metadados_json = json.dumps(metadados, default=Utils.converter_para_json_serializavel)

            with self._conexao() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO previsoes 
                       (criado_em, predicao, probabilidade, metadados_json) 
                       VALUES (?,?,?,?)""",
                    (timestamp, predicao, probabilidade, metadados_json)
                )
                conn.commit()

            logger.info(f"Previsão salva - ID: {cursor.lastrowid}, Predição: {predicao}")

        except Exception as e:
            logger.error(f"Erro ao salvar previsão: {e}")
            raise
