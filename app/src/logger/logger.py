"""
Sistema de logging unificado para toda a aplicação.
"""

import logging
import sys
from typing import Optional

from src.config.params import Params


class Logger:
    """Logger centralizado para toda a aplicação."""

    _instance: Optional['Logger'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._inicializar_logger()
        return cls._instance

    def _inicializar_logger(self):
        """Configura o logger com as configurações padrão."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, Params.LOG_LEVEL))

        # Remove handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Configura formatação
        formatter = logging.Formatter(
            Params.LOG_FORMAT,
            datefmt=Params.LOG_DATE_FORMAT
        )

        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def debug(self, mensagem: str):
        """Log nível DEBUG."""
        self.logger.debug(mensagem)

    def info(self, mensagem: str):
        """Log nível INFO."""
        self.logger.info(mensagem)

    def warning(self, mensagem: str):
        """Log nível WARNING."""
        self.logger.warning(mensagem)

    def error(self, mensagem: str):
        """Log nível ERROR."""
        self.logger.error(mensagem)

    def critical(self, mensagem: str):
        """Log nível CRITICAL."""
        self.logger.critical(mensagem)

    def exception(self, mensagem: str):
        """Log de exceção com traceback."""
        self.logger.exception(mensagem)


# Instância global do logger
logger = Logger()
