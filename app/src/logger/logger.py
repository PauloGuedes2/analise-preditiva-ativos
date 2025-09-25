import logging
import sys
from src.config.params import Params

def get_logger():
    """
    Configura e retorna um logger padrão do Python, garantindo que os handlers
    não sejam duplicados. Esta é a abordagem recomendada e thread-safe.
    """
    logger = logging.getLogger("trading_app_logger")

    # Evita adicionar handlers se o logger já estiver configurado
    if not logger.handlers:
        logger.setLevel(getattr(logging, Params.LOG_LEVEL, logging.INFO))

        formatter = logging.Formatter(
            fmt=Params.LOG_FORMAT,
            datefmt=Params.LOG_DATE_FORMAT
        )

        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Evita que o log seja propagado para o logger root
        logger.propagate = False

    return logger

# Instância global para ser importada por outros módulos
logger = get_logger()