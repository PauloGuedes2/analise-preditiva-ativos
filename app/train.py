import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from joblib import dump

from src.config.params import Params
from src.data.data_loader import DataLoader
from src.logger.logger import logger
from src.models.classification import ClassificadorTrading
from src.models.feature_engineer import FeatureEngineer
from src.models.validation import PurgedKFoldCV
from src.backtesting.risk_analyzer import RiskAnalyzer


class TreinadorModelos:
    """Gerencia o treinamento e salvamento de modelos de trading com validação temporal robusta."""

    def __init__(self):
        """Inicializa o treinador, carregando configurações de tickers e diretórios."""
        self.tickers = Params.TICKERS
        self.diretorio_modelos = Params.PATH_MODELOS

    def _criar_diretorio_modelos(self) -> None:
        """Cria o diretório para salvar os modelos, se não existir."""
        os.makedirs(self.diretorio_modelos, exist_ok=True)
        logger.info(f"Diretório de modelos verificado/criado: '{self.diretorio_modelos}'")

    @staticmethod
    def _validar_dados(df: pd.DataFrame, ticker: str) -> bool:
        """
        Valida se o DataFrame de dados é suficiente para o treinamento.

        Args:
            df (pd.DataFrame): DataFrame com os dados do ativo.
            ticker (str): Nome do ticker para logging.

        Returns:
            bool: True se os dados forem válidos, False caso contrário.
        """
        # Verifica se o número de registros atende ao MINIMO_DADOS_TREINO
        if df.shape[0] < Params.MINIMO_DADOS_TREINO:
            logger.warning(
                f"Dados insuficientes para {ticker}: {df.shape[0]} registros. Mínimo: {Params.MINIMO_DADOS_TREINO}")
            return False
        # Verifica se a quantidade de dados faltantes não excede 5% do total
        if df.isnull().sum().sum() > df.shape[0] * 0.05:
            logger.warning(f"Excesso de dados faltantes para {ticker}: {df.isnull().sum().sum()} valores nulos.")
            return False
        return True

    @staticmethod
    def _realizar_walk_forward_validation(X: pd.DataFrame, y: pd.Series, precos: pd.Series,
                                          t1: pd.Series, ticker: str) -> Dict[str, Any]:
        """Realiza validação walk-forward para estimar performance real."""
        logger.info(f"Iniciando walk-forward validation para {ticker}...")

        cv_gen = PurgedKFoldCV(n_splits=5, t1=t1, purge_days=Params.PURGE_DAYS)
        f1_scores = []
        sharpe_scores = []
        trades_count = []

        for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X)):
            if len(test_idx) == 0:
                continue # Pula folds sem dados de teste

            logger.info(f"Fold {fold + 1}: Treino={len(train_idx)}, Teste={len(test_idx)}")

            # Divide os dados para o fold atual
            modelo_fold = ClassificadorTrading()
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            precos_test = precos.iloc[test_idx]

            # Treina um modelo temporário apenas com os dados do fold
            metricas = modelo_fold.treinar(X_train, y_train, precos.iloc[train_idx], t1.iloc[train_idx])

            if not metricas:
                continue # Pula se o treinamento do fold falhar

            # Gera sinais e calcula performance no conjunto de teste
            df_sinais_test = modelo_fold.prever_e_gerar_sinais(X_test, precos_test, ticker)
            backtest_results = RiskAnalyzer().backtest_sinais(df_sinais_test)

            # Armazena as métricas do fold
            f1_scores.append(metricas.get('f1_macro', 0))
            sharpe_scores.append(backtest_results.get('sharpe', 0))
            trades_count.append(backtest_results.get('trades', 0))

        # Retorna a média das métricas de todos os folds
        return {
            'f1_macro_medio': np.mean(f1_scores) if f1_scores else 0,
            'sharpe_medio': np.mean(sharpe_scores) if sharpe_scores else 0,
            'trades_medio': np.mean(trades_count) if trades_count else 0,
            'folds_validos': len(f1_scores)
        }

    def _treinar_modelo_para_ticker(self, ticker: str) -> bool:
        """Pipeline de treinamento para um único ticker com validação robusta."""
        try:
            # Carrega e valida os dados históricos
            loader = DataLoader()
            df_ohlc, df_ibov = loader.baixar_dados_yf(ticker, periodo=Params.PERIODO_DADOS)

            if not self._validar_dados(df_ohlc, ticker):
                return False

            # Cria features e o dataset final
            feature_engineer = FeatureEngineer()
            X, y, precos, t1, _ = feature_engineer.preparar_dataset(df_ohlc, df_ibov, ticker)
            if X.empty or y.empty:
                logger.error(f"Dataset vazio para {ticker} após engenharia de features.")
                return False

            # Estima a performance real do modelo usando Walk-Forward Validation
            wfv_results = self._realizar_walk_forward_validation(X, y, precos, t1, ticker)
            if wfv_results['folds_validos'] < 3:
                logger.warning(f"{ticker} - Walk-forward validation insuficiente: {wfv_results['folds_validos']} folds")
                return False

            # Treina o modelo final com todos os dados
            modelo = ClassificadorTrading()
            metricas = modelo.treinar(X, y, precos, t1)

            # Critérios de salvamento do modelo
            f1_wfv = wfv_results['f1_macro_medio']
            sharpe_wfv = wfv_results['sharpe_medio']
            trades_wfv = wfv_results['trades_medio']

            # Define os critérios mínimos de performance para salvar o modelo
            criterio_f1 = f1_wfv > 0.50
            criterio_sharpe = sharpe_wfv > -0.1
            criterio_trades = trades_wfv >= 2.5

            logger.info(f"{ticker} - WFV: F1={f1_wfv:.3f}, Sharpe={sharpe_wfv:.3f}, Trades={trades_wfv:.1f}")

            if 'modelo' in locals():
                modelo.wfv_metrics = wfv_results # Anexa as métricas de validação ao objeto do modelo

            # Salva o modelo apenas se todos os critérios forem atendidos
            if criterio_f1 and criterio_sharpe and criterio_trades:
                caminho_modelo = os.path.join(self.diretorio_modelos, f"modelo_{ticker}.joblib")
                dump(modelo, caminho_modelo)
                logger.info(f"✅ {ticker} - Modelo salvo!")
                return True
            else:
                logger.warning(
                    f"❌ {ticker} - Performance insuficiente no WFV, mesmo com critérios flexibilizados."
                )
                return False

        except Exception as e:
            logger.exception(f"❌ Erro crítico ao treinar {ticker}: {e}")
            return False

    def executar_treinamento_completo(self) -> None:
        """Executa o processo de treinamento para todos os tickers configurados."""
        logger.info("🎯 Iniciando processo de treinamento de modelos com validação walk-forward...")
        self._criar_diretorio_modelos()

        tempo_inicio = datetime.now()
        resultados = {ticker: self._treinar_modelo_para_ticker(ticker) for ticker in self.tickers}
        tempo_total = datetime.now() - tempo_inicio

        # Log do relatório final
        modelos_sucesso = sum(1 for sucesso in resultados.values() if sucesso)
        logger.info("=" * 50)
        logger.info("📋 PROCESSO DE TREINAMENTO CONCLUÍDO 📋")
        logger.info(f"⏱️  Tempo total: {tempo_total}")
        logger.info(f"✅ Modelos treinados com sucesso: {modelos_sucesso}/{len(self.tickers)}")
        logger.info("📝 Resultados por ticker:")
        for ticker, sucesso in resultados.items():
            status = "✅ Sucesso" if sucesso else "❌ Falha"
            logger.info(f"      {ticker}: {status}")
        logger.info("=" * 50)

        # Avisos sobre o resultado geral
        if modelos_sucesso == 0:
            logger.warning("⚠️ Nenhum modelo atingiu os critérios de performance para ser salvo.")
        elif modelos_sucesso < len(self.tickers):
            logger.warning(f"⚠️ Apenas {modelos_sucesso} de {len(self.tickers)} modelos foram treinados com sucesso.")
        else:
            logger.info("🎉 Todos os modelos foram treinados com sucesso!")


def main() -> None:
    """Função principal para iniciar o treinamento."""
    treinador = TreinadorModelos()
    treinador.executar_treinamento_completo()


if __name__ == "__main__":
    main()
