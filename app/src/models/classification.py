import os

os.environ['LIGHTGBM_VERBOSE'] = '-1'  # Suprime logs verbosos do LightGBM

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
import optuna
import shap

from src.config.params import Params
from src.logger.logger import logger
from src.utils.utils import Utils
from src.models.validation import PurgedKFoldCV
from src.utils.risk_analyzer import RiskAnalyzer

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suprime logs verbosos do Optuna


class ClassificadorTrading:
    """Encapsula todo o pipeline de um modelo de classificação"""

    def __init__(self):
        """Inicializa o classificador com seus componentes e parâmetros."""
        self.random_state = Params.RANDOM_STATE
        self.n_features = Params.N_FEATURES_A_SELECIONAR
        self.modelo_final = None
        self.features_selecionadas = []
        self.scaler = RobustScaler()  # Scaler robusto a outliers
        self.label_encoder = LabelEncoder()  # Para converter labels {-1, 0, 1} para {0, 1, 2}
        self.threshold_operacional = 0.5  # Threshold de probabilidade para operar
        self.wfv_metrics = {}  # Armazena métricas da validação walk-forward
        self.cv_gen = None  # Gerador de folds da validação cruzada
        self.X_scaled = None  # Dados de treino escalados
        self.training_data_profile = None  # Perfil estatístico dos dados de treino
        self.shap_explainer = None  # Objeto para explicabilidade do modelo

    def treinar(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series, t1: pd.Series) -> Dict[str, Any]:
        """
        Executa o pipeline completo de treinamento: seleção de features, otimização e treino final.

        Args:
            X (pd.DataFrame): DataFrame com as features.
            y (pd.Series): Series com os labels {-1, 0, 1}.
            precos (pd.Series): Series com os preços de fechamento.
            t1 (pd.Series): Series com os timestamps de término de cada evento da tripla barreira.

        Returns:
            Dict[str, Any]: Dicionário com as métricas de performance do modelo treinado.
        """
        logger.info("Iniciando pipeline de treinamento do modelo multiclasse...")
        if not Utils.validar_dados_treinamento(X, y, Params.MINIMO_DADOS_TREINO):
            raise ValueError("Dados de treinamento inválidos ou insuficientes.")

        # Prepara os dados
        y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
        self.features_selecionadas = self._selecionar_features(X, y_encoded)
        if not self.features_selecionadas:
            logger.error("Nenhuma feature selecionada - abortando treinamento")
            return {}

        X_selecionado = X[self.features_selecionadas]
        self.scaler.fit(X_selecionado)
        X_scaled = pd.DataFrame(self.scaler.transform(X_selecionado), index=X_selecionado.index,
                                columns=self.features_selecionadas)

        # Otimiza os hiperparâmetros
        cv_gen = PurgedKFoldCV(n_splits=Params.N_SPLITS_CV, t1=t1, purge_days=Params.PURGE_DAYS)
        best_params = self._otimizar_com_optuna(X_scaled, y_encoded, precos, cv_gen)

        # Treina o modelo final
        final_params = {'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt', 'n_estimators': 1000,
                        'random_state': self.random_state, 'n_jobs': -1, 'verbose': -1, **best_params}
        logger.info("Treinando modelo final com os melhores parâmetros em todos os dados...")
        self.modelo_final = lgb.LGBMClassifier(**final_params, class_weight='balanced')
        self.modelo_final.fit(X_scaled, y_encoded)

        # Gera artefatos pós-treinamento
        logger.info("Criando perfil de dados de treino e explainer SHAP...")
        self.training_data_profile = X_scaled.describe().to_dict()
        self.shap_explainer = shap.TreeExplainer(self.modelo_final)

        all_splits = list(cv_gen.split(X_scaled))
        if not all_splits:
            logger.error("Não foi possível criar splits de validação cruzada.")
            return {}

        # Avalia e calibra o modelo
        _, test_idx = list(cv_gen.split(X_scaled))[-1]
        metricas = self._avaliar_performance(X_scaled.iloc[test_idx], y.iloc[test_idx])
        self.threshold_operacional = self._calibrar_threshold(X_scaled, y_encoded, cv_gen)
        logger.info(f"Threshold operacional calibrado para: {self.threshold_operacional:.3f}")

        self.cv_gen, self.X_scaled = cv_gen, X_scaled
        return metricas

    def _selecionar_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Seleciona as features mais importantes usando um modelo base."""
        logger.info(f"Iniciando seleção de {self.n_features} features de um total de {X.shape[1]}...")
        modelo_base = lgb.LGBMClassifier(random_state=self.random_state, class_weight='balanced', verbose=-1)
        num_features_a_selecionar = min(self.n_features, X.shape[1])
        seletor = SelectFromModel(modelo_base, max_features=num_features_a_selecionar, threshold=-np.inf)
        scaler_temp = RobustScaler()
        X_scaled_temp = scaler_temp.fit_transform(X)
        seletor.fit(X_scaled_temp, y)
        features_selecionadas = X.columns[seletor.get_support()].tolist()
        logger.info(f"Features selecionadas: {len(features_selecionadas)}")
        return features_selecionadas

    def _otimizar_com_optuna(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series, cv_gen) -> Dict[str, Any]:
        """Otimiza os hiperparâmetros do modelo usando Optuna focado em maximizar o Sharpe Ratio."""
        logger.info("Iniciando otimização com Optuna focada em Sharpe Ratio...")
        risk_analyzer = RiskAnalyzer()

        def objective(trial: optuna.Trial) -> float:
            params = {
                'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 150, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 32),
                'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 10.0, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 150),
                'random_state': self.random_state, 'n_jobs': -1, 'verbose': -1,
            }
            threshold = trial.suggest_float('threshold', 0.45, 0.65)
            sharpe_scores = []
            for train_idx, val_idx in cv_gen.split(X):
                if len(train_idx) < 100 or len(val_idx) < 20: continue
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val_enc = y.iloc[train_idx], y.iloc[val_idx]
                precos_val = precos.iloc[val_idx]
                model = lgb.LGBMClassifier(**params, class_weight='balanced')
                model.fit(X_train, y_train, eval_set=[(X_val, y_val_enc)], eval_metric='multi_logloss',
                          callbacks=[lgb.early_stopping(50, verbose=False)])
                idx_classe_1 = np.where(self.label_encoder.classes_ == 1)[0][0]
                probas_val = model.predict_proba(X_val)[:, idx_classe_1]
                sinais = (probas_val > threshold).astype(int)
                df_sinais = pd.DataFrame({'preco': precos_val.values, 'sinal': sinais}, index=precos_val.index)
                backtest_results = risk_analyzer.backtest_sinais(df_sinais, verbose=False)
                sharpe_scores.append(backtest_results.get('sharpe', -1.0))
            return np.mean(sharpe_scores) if sharpe_scores else -1.0

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=Params.OPTUNA_N_TRIALS, timeout=Params.OPTUNA_TIMEOUT_SECONDS)
        logger.info(f"Melhor Sharpe Ratio da otimização: {study.best_value:.4f}")
        return study.best_params

    def _avaliar_performance(self, X_test: pd.DataFrame, y_test_orig: pd.Series) -> Dict[str, Any]:
        """Avalia a performance do modelo final em dados de teste."""
        preds_encoded = self.modelo_final.predict(X_test)
        preds_orig = self.label_encoder.inverse_transform(preds_encoded)
        acuracia = accuracy_score(y_test_orig, preds_orig)
        f1_macro = f1_score(y_test_orig, preds_orig, average='macro', zero_division=0)
        metricas = {'acuracia': acuracia, 'f1_macro': f1_macro}
        logger.info(f"Performance no teste - Acurácia: {acuracia:.3f}, F1-Macro: {f1_macro:.3f}")
        return metricas

    def _calibrar_threshold(self, X: pd.DataFrame, y_enc: pd.Series, cv_gen: PurgedKFoldCV) -> float:
        """Calibra o threshold de decisão para maximizar o F1-Score da classe positiva."""
        thresholds = []
        for train_idx, val_idx in cv_gen.split(X, y_enc):
            probas = self.predict_proba(X.iloc[val_idx])
            y_binary = (self.label_encoder.inverse_transform(y_enc.iloc[val_idx]) == 1).astype(int)
            best_f1, best_thr = 0, 0.5
            for thr in np.arange(0.2, 0.6, 0.01):
                preds = (probas > thr).astype(int)
                f1 = f1_score(y_binary, preds, zero_division=0)
                if f1 > best_f1: best_f1, best_thr = f1, thr
            thresholds.append(best_thr)
        return float(np.mean(thresholds)) if thresholds else 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna as probabilidades previstas para a classe positiva"""
        if self.modelo_final is None: raise RuntimeError("O modelo não foi treinado.")
        if isinstance(X, np.ndarray): X = pd.DataFrame(X, columns=self.features_selecionadas)
        X_scaled = self.scaler.transform(X[self.features_selecionadas])
        idx_classe_1 = np.where(self.label_encoder.classes_ == 1)[0][0]
        return self.modelo_final.predict_proba(X_scaled)[:, idx_classe_1]

    def prever_direcao(self, X_novo: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Faz uma previsão de direção para um novo conjunto de dados."""
        try:
            proba = self.predict_proba(X_novo.tail(1))[-1]
            predicao = 1 if proba >= self.threshold_operacional else 0
            should_operate = bool(predicao == 1)
            return {'probabilidade': float(proba), 'predicao': int(predicao), 'should_operate': should_operate,
                    'threshold_operacional': float(self.threshold_operacional), 'status': 'sucesso'}
        except Exception as e:
            logger.error(f"Erro ao prever direção: {e}")
            return {'status': f'erro: {str(e)}', 'probabilidade': 0.5, 'predicao': 0, 'should_operate': False}

    def prever_e_gerar_sinais(self, X: pd.DataFrame, precos: pd.Series, ticker: str,
                              threshold_override: float = None) -> pd.DataFrame:
        """Gera sinais de compra/venda para um conjunto de dados, com threshold customizável."""
        threshold = threshold_override if threshold_override is not None else self.threshold_operacional
        probas = self.predict_proba(X)
        sinais = (probas >= threshold).astype(int)
        return pd.DataFrame({'preco': precos.values, 'sinal': sinais}, index=precos.index)
