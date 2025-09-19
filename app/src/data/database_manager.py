import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from src.config.params import Params
from src.logger.logger import logger
from src.utils.utils import Utils


class DatabaseManager:
    """Gerencia operações de banco de dados SQLite para metadados do modelo."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or Params.PATH_DB_METADATA
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
        """Cria tabelas necessárias se não existirem."""
        try:
            with self._conexao() as conn:
                cursor = conn.cursor()

                # Tabela de metadados de treino
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS treino_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        criado_em TEXT,
                        metadata_json TEXT
                    )
                """)

                # Tabela de previsões
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
        """Retorna timestamp atual em formato ISO."""
        return datetime.utcnow().isoformat()

    def salvar_treino_metadata(self, metadata: Dict[str, Any]):
        """Salva metadados de treino no banco."""
        try:
            timestamp = self._obter_timestamp_atual()
            metadata_json = json.dumps(metadata, default=Utils.converter_para_json_serializavel)

            with self._conexao() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO treino_metadata (criado_em, metadata_json) VALUES (?,?)",
                    (timestamp, metadata_json)
                )
                conn.commit()

            logger.info(f"Metadados de treino salvos - ID: {cursor.lastrowid}")

        except Exception as e:
            logger.error(f"Erro ao salvar metadados de treino: {e}")
            raise

    def salvar_previsao(self, dados: Dict[str, Any]):
        """Salva dados de previsão no banco."""
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

    def buscar_ultimo_treino(self) -> Optional[Dict[str, Any]]:
        """Busca o último registro de treino do banco."""
        try:
            with self._conexao() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT metadata_json FROM treino_metadata ORDER BY id DESC LIMIT 1"
                )
                resultado = cursor.fetchone()

                if resultado:
                    metadata = json.loads(resultado[0])
                    logger.debug("Último treino recuperado do banco")
                    return metadata

                logger.warning("Nenhum registro de treino encontrado")
                return None

        except Exception as e:
            logger.error(f"Erro ao buscar último treino: {e}")
            return None

    def buscar_previsoes_recentes(self, limite: int = None) -> list:
        """Busca as previsões mais recentes do banco."""
        limite = limite or Params.LIMITE_PREVISOES_RECENTES

        try:
            with self._conexao() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT criado_em, predicao, probabilidade, metadados_json 
                       FROM previsoes ORDER BY id DESC LIMIT ?""",
                    (limite,)
                )

                resultados = []
                for row in cursor.fetchall():
                    resultados.append({
                        'criado_em': row[0],
                        'predicao': row[1],
                        'probabilidade': row[2],
                        'metadados': json.loads(row[3])
                    })

                logger.debug(f"{len(resultados)} previsões recentes recuperadas")
                return resultados

        except Exception as e:
            logger.error(f"Erro ao buscar previsões recentes: {e}")
            return []
