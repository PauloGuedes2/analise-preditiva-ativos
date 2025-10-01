import os

os.environ['LIGHTGBM_VERBOSE'] = '-1'  # Suprime logs verbosos do LightGBM

from typing import Dict, Any, List, Tuple, Optional
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
from src.models.validation import PurgedKFoldCV
from src.backtesting.risk_analyzer import RiskAnalyzer

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suprime logs verbosos do Optuna


class ClassificadorTrading:
    """Encapsula todo o pipeline de um modelo de classificação."""

    def __init__(self):
        """Inicializa o classificador com seus componentes e parâmetros."""
        self.random_state = Params.RANDOM_STATE
        self.n_features = Params.N_FEATURES_A_SELECIONAR
        self.modelo_final = None
        self.features_selecionadas = []
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.threshold_operacional = 0.5
        self.wfv_metrics = {}
        self.cv_gen = None
        self.X_scaled = None
        self.training_data_profile = None
        self.shap_explainer = None

    def treinar(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series, t1: pd.Series) -> Dict[str, Any]:
        """Orquestra o pipeline completo de treinamento do modelo."""
        logger.info("Iniciando pipeline de treinamento do modelo multiclasse...")
        if not self.validar_dados_treinamento(X, y, Params.MINIMO_DADOS_TREINO):
            raise ValueError("Dados de treinamento inválidos ou insuficientes.")

        # Etapa 1: Preparar dados, selecionar features e escalar
        X_scaled, y_encoded = self._preparar_dados_para_treino(X, y)
        if X_scaled is None:
            return {}

        # Etapa 2: Otimizar hiperparâmetros
        cv_gen = PurgedKFoldCV(n_splits=Params.N_SPLITS_CV, t1=t1, purge_days=Params.PURGE_DAYS)
        best_params = self._otimizar_com_optuna(X_scaled, y_encoded, precos, cv_gen)

        # Etapa 3: Treinar o modelo final com os melhores parâmetros
        self._treinar_modelo_final(X_scaled, y_encoded, best_params)

        # Etapa 4: Gerar artefatos pós-treinamento para análise
        self._gerar_artefatos_pos_treino(X_scaled)

        # Etapa 5: Avaliar performance e calibrar o threshold de decisão
        _, test_idx = list(cv_gen.split(X_scaled))[-1]
        metricas = self._avaliar_performance(X_scaled.iloc[test_idx], y.iloc[test_idx])
        self.threshold_operacional = self._calibrar_threshold(X_scaled, y_encoded, cv_gen)
        logger.info(f"Threshold operacional calibrado para: {self.threshold_operacional:.3f}")

        # Salva objetos para uso posterior (ex: matriz de confusão na UI)
        self.cv_gen, self.X_scaled = cv_gen, X_scaled
        return metricas

    def _preparar_dados_para_treino(self, X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Codifica labels, seleciona features e escala os dados."""
        y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index)

        self.features_selecionadas = self._selecionar_features(X, y_encoded)
        if not self.features_selecionadas:
            logger.error("Nenhuma feature selecionada - abortando treinamento.")
            return None, None

        X_selecionado = X[self.features_selecionadas]
        self.scaler.fit(X_selecionado)
        X_scaled = pd.DataFrame(self.scaler.transform(X_selecionado), index=X_selecionado.index,
                                columns=self.features_selecionadas)
        return X_scaled, y_encoded

    def _treinar_modelo_final(self, X_scaled: pd.DataFrame, y_encoded: pd.Series, best_params: Dict[str, Any]):
        """Treina o modelo final com os melhores parâmetros encontrados."""
        final_params = {'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt', 'n_estimators': 1000,
                        'random_state': self.random_state, 'n_jobs': -1, 'verbose': -1, **best_params}
        logger.info("Treinando modelo final com os melhores parâmetros em todos os dados...")
        self.modelo_final = lgb.LGBMClassifier(**final_params, class_weight='balanced')
        self.modelo_final.fit(X_scaled, y_encoded)

    def _gerar_artefatos_pos_treino(self, X_scaled: pd.DataFrame):
        """Cria o perfil dos dados de treino e o explainer SHAP para uso futuro."""
        logger.info("Criando perfil de dados de treino e explainer SHAP...")
        self.training_data_profile = X_scaled.describe().to_dict()
        try:
            self.shap_explainer = shap.TreeExplainer(self.modelo_final)
        except Exception as e:
            logger.warning(f"Não foi possível criar o explainer SHAP: {e}")
            self.shap_explainer = None

    def _otimizar_com_optuna(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series, cv_gen) -> Dict[str, Any]:
        """Configura e executa o estudo de otimização com Optuna."""
        logger.info("Iniciando otimização com Optuna focada em Sharpe Ratio...")

        objective_func = lambda trial: self._objetivo_optuna(trial, X, y, precos, cv_gen)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective_func, n_trials=Params.OPTUNA_N_TRIALS, timeout=Params.OPTUNA_TIMEOUT_SECONDS)

        logger.info(f"Melhor Sharpe Ratio da otimização: {study.best_value:.4f}")
        # Remove o threshold do dicionário de parâmetros do modelo
        best_params = study.best_params
        best_params.pop('threshold', None)

        return best_params

    def _objetivo_optuna(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, precos: pd.Series, cv_gen) -> float:
        """Função objetivo que o Optuna tenta maximizar."""
        risk_analyzer = RiskAnalyzer()
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


    def gerar_performance_wfv_agregada(self, y: pd.Series, precos: pd.Series, t1: pd.Series) -> dict:
        """
        Usa os dados já processados do modelo treinado para agregar os retornos de todos
        os folds de teste de forma consistente, garantindo o alinhamento dos índices.
        """
        logger.info("Gerando performance agregada da Validação Walk-Forward...")
        risk_analyzer = RiskAnalyzer()

        if self.X_scaled is None:
            logger.error("Atributo X_scaled não definido. Rode .treinar() primeiro.")
            return risk_analyzer.retornar_metricas_vazias()

        common_index = self.X_scaled.index.intersection(y.index).intersection(precos.index).intersection(t1.index)

        X_aligned = self.X_scaled.loc[common_index]
        y_aligned = y.loc[common_index]
        precos_aligned = precos.loc[common_index]
        t1_aligned = t1.loc[common_index]

        y_encoded_aligned = pd.Series(self.label_encoder.transform(y_aligned), index=common_index)
        cv_gen = PurgedKFoldCV(n_splits=Params.N_SPLITS_CV, t1=t1_aligned, purge_days=Params.PURGE_DAYS)

        todos_os_retornos = []
        todos_os_sinais_wfv = []

        for train_idx, test_idx in cv_gen.split(X_aligned):
            if len(test_idx) == 0: continue

            X_train, X_test = X_aligned.iloc[train_idx], X_aligned.iloc[test_idx]
            y_train_enc = y_encoded_aligned.iloc[train_idx]
            precos_test = precos_aligned.iloc[test_idx]

            modelo_fold = lgb.LGBMClassifier(**self.modelo_final.get_params())
            modelo_fold.fit(X_train, y_train_enc)

            idx_classe_1 = np.where(self.label_encoder.classes_ == 1)[0][0]
            probas_test = modelo_fold.predict_proba(X_test)[:, idx_classe_1]
            sinais = (probas_test >= self.threshold_operacional).astype(int)

            df_sinais_test = pd.DataFrame({'preco': precos_test.values, 'sinal': sinais}, index=precos_test.index)

            sinais_positivos = df_sinais_test[df_sinais_test['sinal'] == 1]
            for data, linha in sinais_positivos.iterrows():
                todos_os_sinais_wfv.append({'data': data, 'preco': linha['preco']})

            backtest_fold = risk_analyzer.backtest_sinais(df_sinais_test, verbose=False)

            if backtest_fold['trades'] > 0:
                todos_os_retornos.extend(backtest_fold['retornos'])

        if not todos_os_retornos:
            return risk_analyzer.retornar_metricas_vazias()

        retornos_np = np.array(todos_os_retornos)
        capital_total = np.insert(np.cumprod(1 + retornos_np), 0, 1)
        pico = np.maximum.accumulate(capital_total)
        drawdown_series = (capital_total - pico) / pico

        return {
            'retorno_total': float(capital_total[-1] - 1),
            'trades': len(retornos_np),
            'sharpe': self.wfv_metrics.get('sharpe_medio', 0),
            'max_drawdown': float(np.min(drawdown_series)),
            'win_rate': np.sum(retornos_np > 0) / len(retornos_np),
            'equity_curve': capital_total.tolist(),
            'retornos': retornos_np.tolist(),
            'sinais_wfv': todos_os_sinais_wfv,
        }

    @staticmethod
    def validar_dados_treinamento(X: pd.DataFrame, y: pd.Series, min_amostras: int = 100) -> bool:
        """Valida dados para treinamento."""
        if len(X) < min_amostras or len(y) < min_amostras: return False
        if len(X) != len(y): return False
        if y.nunique() < 2: return False
        return True

    def _selecionar_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Seleciona as features mais importantes usando um modelo base."""
        num_features_disponiveis = X.shape[1]
        num_features_a_selecionar = min(self.n_features, num_features_disponiveis)
        logger.info(
            f"Iniciando seleção de {num_features_a_selecionar} features de um total de {num_features_disponiveis}...")

        modelo_base = lgb.LGBMClassifier(random_state=self.random_state, class_weight='balanced', verbose=-1)
        seletor = SelectFromModel(modelo_base, max_features=num_features_a_selecionar, threshold=-np.inf)

        scaler_temp = RobustScaler()
        X_scaled_temp = scaler_temp.fit_transform(X)
        seletor.fit(X_scaled_temp, y)

        features_selecionadas = X.columns[seletor.get_support()].tolist()
        logger.info(f"Features selecionadas: {len(features_selecionadas)}")
        return features_selecionadas

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
            if len(val_idx) == 0: continue
            probas = self.predict_proba(X.iloc[val_idx])
            y_binary = (self.label_encoder.inverse_transform(y_enc.iloc[val_idx]) == 1).astype(int)
            best_f1, best_thr = 0, 0.5
            for thr in np.arange(0.2, 0.6, 0.01):
                preds = (probas > thr).astype(int)
                f1 = f1_score(y_binary, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            thresholds.append(best_thr)
        return float(np.mean(thresholds)) if thresholds else 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna as probabilidades previstas para a classe positiva (label=1)."""
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
            logger.error(f"Erro ao prever direção para {ticker}: {e}")
            return {'status': f'erro: {str(e)}', 'probabilidade': 0.5, 'predicao': 0, 'should_operate': False}

    def prever_e_gerar_sinais(self, X: pd.DataFrame, precos: pd.Series, ticker: str,
                              threshold_override: float = None) -> pd.DataFrame:
        """Gera sinais de compra/venda para um conjunto de dados, com threshold customizável."""
        threshold = threshold_override if threshold_override is not None else self.threshold_operacional
        probas = self.predict_proba(X)
        sinais = (probas >= threshold).astype(int)
        return pd.DataFrame({'preco': precos.values, 'sinal': sinais}, index=precos.index)
