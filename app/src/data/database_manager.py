import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager


class DatabaseManager:
    """Gerencia operações de banco de dados SQLite para metadados do modelo."""

    def __init__(self, db_path: str = 'model_metadata.db'):
        self.db_path = db_path
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

    @staticmethod
    def _obter_timestamp_atual() -> str:
        """Retorna timestamp atual em formato ISO."""
        return datetime.utcnow().isoformat()

    def salvar_treino_metadata(self, metadata: Dict[str, Any]):
        """Salva metadados de treino no banco."""
        timestamp = self._obter_timestamp_atual()
        metadata_json = json.dumps(metadata)

        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO treino_metadata (criado_em, metadata_json) VALUES (?,?)",
                (timestamp, metadata_json)
            )
            conn.commit()

    def salvar_previsao(self, dados: Dict[str, Any]):
        """Salva dados de previsão no banco."""
        timestamp = self._obter_timestamp_atual()
        predicao = dados.get('predicao')
        probabilidade = dados.get('probabilidade')
        metadados = dados.get('metadados', {})
        metadados_json = json.dumps(metadados)

        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO previsoes 
                   (criado_em, predicao, probabilidade, metadados_json) 
                   VALUES (?,?,?,?)""",
                (timestamp, predicao, probabilidade, metadados_json)
            )
            conn.commit()

    def buscar_ultimo_treino(self) -> Optional[Dict[str, Any]]:
        """Busca o último registro de treino do banco."""
        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT metadata_json FROM treino_metadata ORDER BY id DESC LIMIT 1"
            )
            resultado = cursor.fetchone()

            if resultado:
                return json.loads(resultado[0])
            return None

    def buscar_previsoes_recentes(self, limite: int = 10) -> list:
        """Busca as previsões mais recentes do banco."""
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

            return resultados