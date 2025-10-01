import threading
import time
from datetime import datetime
from typing import Dict, List

from src.data.data_loader import DataLoader
from src.logger.logger import logger


class DataUpdater:
    """Serviço singleton para atualizar dados de mercado em uma thread de background."""

    _instance = None
    _lock = threading.Lock() # Garante que a criação da instância seja thread-safe

    def __new__(cls):
        # Implementação do padrão Singleton
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._inicializar()
            return cls._instance

    def _inicializar(self):
        """Inicializador interno para a instância singleton."""
        self.loader = DataLoader()
        self.ultima_atualizacao: Dict[str, datetime] = {}
        self.executando = False
        self.thread = None

    def iniciar_atualizacao_automatica(self, tickers: List[str], intervalo_minutos: int = 30):
        """Inicia o processo de atualização automática em uma thread separada."""
        if self.executando:
            return # Evita iniciar múltiplas threads

        self.executando = True
        self.thread = threading.Thread(
            target=self._loop_atualizacao,
            args=(tickers, intervalo_minutos),
            daemon=True # Permite que o programa principal saia sem esperar a thread
        )
        self.thread.start()
        logger.info(f"Serviço de atualização automática iniciado (intervalo: {intervalo_minutos}min)")

    def _loop_atualizacao(self, tickers: List[str], intervalo_minutos: int):
        """Loop principal que executa a atualização em intervalos definidos."""
        while self.executando:
            try:
                self.atualizar_todos_tickers(tickers)

                # Loop de espera que verifica a cada segundo se deve parar
                for _ in range(intervalo_minutos * 60):
                    if not self.executando:
                        break  # Sai do loop de espera imediatamente se a flag mudar
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Erro no loop de atualização: {e}")
                time.sleep(60)

    def atualizar_todos_tickers(self, tickers: List[str]):
        """Itera sobre a lista de tickers e tenta atualizar cada um."""
        for ticker in tickers:
            try:
                self.atualizar_ticker(ticker)
            except Exception as e:
                logger.error(f"Erro ao atualizar {ticker}: {e}")

    def atualizar_ticker(self, ticker: str) -> bool:
        """
        Atualiza os dados para um ticker específico, se necessário.

        A atualização só ocorre se a última atualização foi há mais de uma hora.
        """
        agora = datetime.now()

        # Limita a frequência de atualização para no máximo uma vez por hora
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
        """Sinaliza para a thread de atualização parar e aguarda sua finalização."""
        self.executando = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
            logger.info("Serviço de atualização parado")

# Instância global
data_updater = DataUpdater()
