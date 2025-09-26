import os
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump

from src.backtesting.risk_analyzer import RiskAnalyzer
from src.config.params import Params
from src.data.data_loader import DataLoader
from src.logger.logger import logger
from src.models.classification import ClassificadorTrading
from src.models.feature_engineer import FeatureEngineer
from src.models.validation import PurgedKFoldCV


class TreinadorModelos:
    """Gerencia o pipeline de treinamento e salvamento de modelos de trading."""

    def __init__(self):
        """Inicializa o treinador, carregando configurações."""
        self.tickers = Params.TICKERS
        self.diretorio_modelos = Params.PATH_MODELOS
        self.risk_analyzer = RiskAnalyzer()

    def executar_treinamento_completo(self) -> None:
        """Executa o processo de treinamento para todos os tickers configurados."""
        logger.info("🤖 Iniciando processo de treinamento de modelos...")
        self._criar_diretorio_modelos()
        tempo_inicio = datetime.now()

        resultados = {ticker: self._treinar_modelo_para_ticker(ticker) for ticker in self.tickers}

        tempo_total = datetime.now() - tempo_inicio
        self._logar_relatorio_final(resultados, tempo_total)

    def _treinar_modelo_para_ticker(self, ticker: str) -> bool:
        """Orquestra o pipeline de treinamento para um único ticker."""
        try:
            # 1. Preparação dos Dados
            dados_preparados = self._preparar_dados_para_treino(ticker)
            if not dados_preparados:
                return False
            X, y, precos, t1 = dados_preparados

            # 2. Validação da Estratégia (Walk-Forward)
            wfv_results = self._realizar_walk_forward_validation(X, y, precos, t1, ticker)
            if wfv_results['folds_validos'] < 3:
                logger.warning(
                    f"{ticker} - WFV insuficiente: {wfv_results['folds_validos']} folds válidos. Modelo não será treinado.")
                return False

            # 3. Treinamento do Modelo Final
            modelo = self._treinar_modelo_final(X, y, precos, t1)
            modelo.wfv_metrics = wfv_results

            # 4. Avaliação e Persistência
            return self._avaliar_e_salvar_modelo(modelo, ticker)

        except Exception as e:
            logger.exception(f"❌ Erro crítico no pipeline de treinamento para {ticker}: {e}")
            return False

    def _preparar_dados_para_treino(self, ticker: str) -> Optional[
        Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]]:
        """Carrega, valida e realiza a engenharia de features."""
        logger.info(f"--- Iniciando Etapa 1: Preparação de Dados para {ticker} ---")
        loader = DataLoader()
        df_ohlc, df_ibov = loader.baixar_dados_yf(ticker, periodo=Params.PERIODO_DADOS)

        if not self._validar_dados(df_ohlc, ticker):
            return None

        feature_engineer = FeatureEngineer()
        X, y, precos, t1, _ = feature_engineer.preparar_dataset(df_ohlc, df_ibov, ticker)

        if X.empty or y.empty:
            logger.error(f"Dataset vazio para {ticker} após engenharia de features.")
            return None

        return X, y, precos, t1

    def _realizar_walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series, t1: pd.Series,
                                          ticker: str) -> Dict[str, Any]:
        """Executa a validação Walk-Forward para estimar a performance real do modelo."""
        logger.info(f"--- Iniciando Etapa 2: Walk-Forward Validation para {ticker} ---")
        cv_gen = PurgedKFoldCV(n_splits=Params.N_SPLITS_CV, t1=t1, purge_days=Params.PURGE_DAYS)
        f1_scores, sharpe_scores, trades_count = [], [], []

        for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X)):
            if len(test_idx) == 0: continue

            logger.info(f"Fold {fold + 1}/{Params.N_SPLITS_CV}: Treino={len(train_idx)}, Teste={len(test_idx)}")
            modelo_fold = ClassificadorTrading()
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            precos_test = precos.iloc[test_idx]

            metricas_treino = modelo_fold.treinar(X_train, y_train, precos.iloc[train_idx], t1.iloc[train_idx])
            if not metricas_treino: continue

            df_sinais_test = modelo_fold.prever_e_gerar_sinais(X_test, precos_test, ticker)
            backtest_results = self.risk_analyzer.backtest_sinais(df_sinais_test, verbose=False)

            f1_scores.append(metricas_treino.get('f1_macro', 0))
            sharpe_scores.append(backtest_results.get('sharpe', 0))
            trades_count.append(backtest_results.get('trades', 0))

        return {
            'f1_macro_medio': np.mean(f1_scores) if f1_scores else 0,
            'sharpe_medio': np.mean(sharpe_scores) if sharpe_scores else 0,
            'trades_medio': np.mean(trades_count) if trades_count else 0,
            'folds_validos': len(f1_scores)
        }

    @staticmethod
    def _treinar_modelo_final(X: pd.DataFrame, y: pd.Series, precos: pd.Series,
                              t1: pd.Series) -> ClassificadorTrading:
        """Instancia e treina o modelo final com todos os dados disponíveis."""
        logger.info("--- Iniciando Etapa 3: Treinamento do Modelo Final ---")
        modelo = ClassificadorTrading()
        modelo.treinar(X, y, precos, t1)
        return modelo

    def _avaliar_e_salvar_modelo(self, modelo: ClassificadorTrading, ticker: str) -> bool:
        """Verifica se o modelo atende aos critérios de performance e, se sim, o salva."""
        logger.info(f"--- Iniciando Etapa 4: Avaliação e Persistência para {ticker} ---")
        wfv_results = modelo.wfv_metrics
        f1_wfv = wfv_results['f1_macro_medio']
        sharpe_wfv = wfv_results['sharpe_medio']
        trades_wfv = wfv_results['trades_medio']

        criterio_f1 = f1_wfv > 0.50
        criterio_sharpe = sharpe_wfv > -0.1
        criterio_trades = trades_wfv >= 2.5

        logger.info(
            f"Critérios de WFV para {ticker}: F1={f1_wfv:.3f}, Sharpe={sharpe_wfv:.3f}, Trades={trades_wfv:.1f}")

        if criterio_f1 and criterio_sharpe and criterio_trades:
            caminho_modelo = os.path.join(self.diretorio_modelos, f"modelo_{ticker}.joblib")
            dump(modelo, caminho_modelo)
            logger.info(f"✅ {ticker} - Modelo atendeu aos critérios e foi salvo!")
            return True
        else:
            logger.warning(f"❌ {ticker} - Performance insuficiente no WFV. Modelo não foi salvo.")
            return False

    def _criar_diretorio_modelos(self) -> None:
        """Cria o diretório para salvar os modelos, se não existir."""
        os.makedirs(self.diretorio_modelos, exist_ok=True)

    @staticmethod
    def _validar_dados(df: pd.DataFrame, ticker: str) -> bool:
        """Valida se o DataFrame de dados é suficiente para o treinamento."""
        if df.shape[0] < Params.MINIMO_DADOS_TREINO:
            logger.warning(
                f"Dados insuficientes para {ticker}: {df.shape[0]} registros. Mínimo: {Params.MINIMO_DADOS_TREINO}")
            return False
        if df.isnull().sum().sum() > df.shape[0] * 0.05:
            logger.warning(f"Excesso de dados faltantes para {ticker}: {df.isnull().sum().sum()} valores nulos.")
            return False
        return True

    def _logar_relatorio_final(self, resultados: Dict[str, bool], tempo_total: timedelta):
        """Loga um sumário completo do processo de treinamento."""
        modelos_sucesso = sum(1 for sucesso in resultados.values() if sucesso)
        logger.info("=" * 60)
        logger.info("📋 PROCESSO DE TREINAMENTO CONCLUÍDO 📋")
        logger.info(f"⏱️  Tempo total de execução: {tempo_total}")
        logger.info(f"✅ Modelos treinados com sucesso: {modelos_sucesso}/{len(self.tickers)}")
        logger.info("📝 Resultados detalhados por ticker:")
        for ticker, sucesso in resultados.items():
            status = "✅ Sucesso" if sucesso else "❌ Falha (não atingiu os critérios)"
            logger.info(f"      - {ticker}: {status}")
        logger.info("=" * 60)
        if modelos_sucesso == 0:
            logger.warning("⚠️ Nenhum modelo atingiu os critérios de performance para ser salvo.")
        elif modelos_sucesso < len(self.tickers):
            logger.warning(f"⚠️ Apenas {modelos_sucesso} de {len(self.tickers)} modelos foram treinados com sucesso.")
        else:
            logger.info("🎉 Todos os modelos foram treinados com sucesso!")
