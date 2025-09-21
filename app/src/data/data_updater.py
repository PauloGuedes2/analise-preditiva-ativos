import threading
import time
from datetime import datetime
from typing import Dict, List

from src.data.data_loader import DataLoader
from src.logger.logger import logger


class DataUpdater:
    """Serviço para atualizar dados em background."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._inicializar()
            return cls._instance

    def _inicializar(self):
        self.loader = DataLoader()
        self.ultima_atualizacao: Dict[str, datetime] = {}
        self.executando = False
        self.thread = None

    def iniciar_atualizacao_automatica(self, tickers: List[str], intervalo_minutos: int = 30):
        """Inicia a atualização automática em background."""
        if self.executando:
            return

        self.executando = True
        self.thread = threading.Thread(
            target=self._loop_atualizacao,
            args=(tickers, intervalo_minutos),
            daemon=True
        )
        self.thread.start()
        logger.info(f"Serviço de atualização automática iniciado (intervalo: {intervalo_minutos}min)")

    def _loop_atualizacao(self, tickers: List[str], intervalo_minutos: int):
        """Loop principal de atualização."""
        while self.executando:
            try:
                self.atualizar_todos_tickers(tickers)
                time.sleep(intervalo_minutos * 60)
            except Exception as e:
                logger.error(f"Erro no loop de atualização: {e}")
                time.sleep(60)  # Espera 1 minuto em caso de erro

    def atualizar_todos_tickers(self, tickers: List[str]):
        """Atualiza todos os tickers da lista."""
        for ticker in tickers:
            try:
                self.atualizar_ticker(ticker)
            except Exception as e:
                logger.error(f"Erro ao atualizar {ticker}: {e}")

    def atualizar_ticker(self, ticker: str) -> bool:
        """Atualiza dados de um ticker específico."""
        agora = datetime.now()

        # Verifica se precisa atualizar (mínimo 1 hora desde última atualização)
        if (ticker in self.ultima_atualizacao and
                (agora - self.ultima_atualizacao[ticker]).total_seconds() < 3600):
            return False

        try:
            df_ticker, df_ibov = self.loader.atualizar_dados_ticker(ticker)
            self.ultima_atualizacao[ticker] = agora

            if not df_ticker.empty:
                logger.info(f"✅ {ticker} atualizado com sucesso")
                return True
            else:
                logger.warning(f"⚠️  {ticker} não foi atualizado (dados vazios)")
                return False

        except Exception as e:
            logger.error(f"❌ Erro ao atualizar {ticker}: {e}")
            raise

    def parar_atualizacao(self):
        """Para o serviço de atualização."""
        self.executando = False
        if self.thread:
            self.thread.join(timeout=5)
            logger.info("Serviço de atualização parado")

    def verificar_necessidade_atualizacao(self, ticker: str) -> bool:
        """Verifica se um ticker precisa ser atualizado."""
        if ticker not in self.ultima_atualizacao:
            return True

        # Atualiza a cada hora no máximo
        return (datetime.now() - self.ultima_atualizacao[ticker]).total_seconds() > 3600


# Instância global
data_updater = DataUpdater()
