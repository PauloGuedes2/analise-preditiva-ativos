"""
Gerenciador SQLite simples para salvar metadados de treino e previsões.
Arquivo: treino_metadata (json), previsoes (cada previsao com prob/pred/metadados)
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any


class DatabaseManagerRefinado:
    def __init__(self, db_path: str = 'model_metadata.db'):
        self.db_path = db_path
        self._criar_tabelas()

    def _con(self):
        return sqlite3.connect(self.db_path)

    def _criar_tabelas(self):
        with self._con() as c:
            cur = c.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS treino_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    criado_em TEXT,
                    metadata_json TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS previsoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    criado_em TEXT,
                    predicao INTEGER,
                    probabilidade REAL,
                    metadados_json TEXT
                )
            """)
            c.commit()

    def salvar_treino_metadata(self, metadata: Dict[str, Any]):
        with self._con() as c:
            cur = c.cursor()
            cur.execute("INSERT INTO treino_metadata (criado_em, metadata_json) VALUES (?,?)",
                        (datetime.utcnow().isoformat(), json.dumps(metadata)))
            c.commit()

    def salvar_previsao(self, dados: Dict[str, Any]):
        criado = datetime.utcnow().isoformat()
        pred = dados.get('predicao')
        prob = dados.get('probabilidade')
        meta = dados.get('metadados', {})
        with self._con() as c:
            cur = c.cursor()
            cur.execute("INSERT INTO previsoes (criado_em, predicao, probabilidade, metadados_json) VALUES (?,?,?,?)",
                        (criado, pred, prob, json.dumps(meta)))
            c.commit()
