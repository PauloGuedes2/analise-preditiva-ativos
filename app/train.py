import os
from datetime import datetime

import pandas as pd
from joblib import dump

from src.data.data_loader import DataLoader
from src.logger.logger import logger
from src.models.classification import ClassificadorTrading
from src.models.feature_engineer import FeatureEngineer


class TreinadorModelos:
    """Gerencia o treinamento e salvamento de modelos de trading."""

    def __init__(self):
        self.tickers = [
            "PETR4.SA", "VALE3.SA", "ITSA4.SA",
            "TAEE11.SA", "BBSE3.SA", "ABEV3.SA"
        ]
        self.diretorio_modelos = "modelos_treinados"

    def _criar_diretorio_modelos(self):
        """Cria diretório para modelos se não existir."""
        os.makedirs(self.diretorio_modelos, exist_ok=True)
        logger.info(f"Diretório de modelos: {self.diretorio_modelos}")

    @staticmethod
    def _validar_dados(df_ohlc: pd.DataFrame, ticker: str) -> bool:
        """
        Valida se os dados são suficientes para treinamento.

        Args:
            df_ohlc: DataFrame com dados OHLC
            ticker: Símbolo do ativo para logging

        Returns:
            True se dados são válidos, False caso contrário
        """
        if len(df_ohlc) < 100:
            logger.warning(
                f"Dados insuficientes para {ticker}: {len(df_ohlc)} registros"
            )
            return False
        return True

    def _treinar_modelo_ticker(self, ticker: str) -> bool:
        """
        Treina modelo para um ticker específico.

        Args:
            ticker: Símbolo do ativo

        Returns:
            True se treinamento bem-sucedido, False caso contrário
        """
        logger.info(f"🚀 Iniciando treinamento para {ticker}")

        try:
            # Carregar dados
            loader = DataLoader()
            df_ohlc, df_ibov = loader.baixar_dados_yf(ticker, periodo="3y")

            # Validar dados
            if not self._validar_dados(df_ohlc, ticker):
                return False

            # Engenharia de features
            feature_engineer = FeatureEngineer()
            X, y, precos = feature_engineer.preparar_dataset_classificacao(df_ohlc, df_ibov)

            # Treinar modelo
            modelo = ClassificadorTrading(n_features=20, random_state=42)
            metricas_treinamento = modelo.treinar(X, y, precos, n_splits=4, purge_days=2)

            # Salvar modelo
            caminho_modelo = os.path.join(self.diretorio_modelos, f"modelo_{ticker}.joblib")
            dump(modelo, caminho_modelo)

            logger.info(
                f"✅ Modelo para {ticker} treinado e salvo em: {caminho_modelo}\n"
                f"   📊 Métricas: {metricas_treinamento}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao treinar {ticker}: {e}")
            return False

    def executar_treinamento(self):
        """Executa treinamento para todos os tickers."""
        logger.info("🎯 Iniciando processo de treinamento de modelos")
        self._criar_diretorio_modelos()

        resultados = []
        tempo_inicio = datetime.now()

        for ticker in self.tickers:
            sucesso = self._treinar_modelo_ticker(ticker)
            resultados.append((ticker, sucesso))

        # Relatório final
        tempo_total = datetime.now() - tempo_inicio
        modelos_sucesso = sum(1 for _, sucesso in resultados if sucesso)

        logger.info(
            f"📋 Processo de treinamento concluído\n"
            f"   ⏱️  Tempo total: {tempo_total}\n"
            f"   ✅ Modelos treinados com sucesso: {modelos_sucesso}/{len(self.tickers)}\n"
            f"   📝 Resultados por ticker:"
        )

        for ticker, sucesso in resultados:
            status = "✅" if sucesso else "❌"
            logger.info(f"      {status} {ticker}")

        if modelos_sucesso == 0:
            logger.warning("⚠️  Nenhum modelo foi treinado com sucesso")
        elif modelos_sucesso < len(self.tickers):
            logger.warning(
                f"⚠️  Apenas {modelos_sucesso} de {len(self.tickers)} "
                f"modelos foram treinados com sucesso"
            )
        else:
            logger.info("🎉 Todos os modelos treinados com sucesso!")


def main():
    """Função principal do script de treinamento."""
    treinador = TreinadorModelos()
    treinador.executar_treinamento()


if __name__ == "__main__":
    main()
