from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.data.database_manager import DatabaseManager
from src.models.feature_engineer import FeatureEngineer
from src.utils.risk_analyzer import RiskAnalyzer


class ClassificadorTrading:
    """Sistema de classificação para previsão de direção de preços."""

    def __init__(self, n_features: int = 25, random_state: int = 42,
                 confidence_operar: float = 0.60, otimizar_hiperparametros: bool = True):

        self.n_features = n_features
        self.random_state = random_state
        self.confidence_operar = confidence_operar
        self.otimizar_hiperparametros = otimizar_hiperparametros

        self._inicializar_modelos_base()
        self._inicializar_utilitarios()

    def _inicializar_modelos_base(self):
        """Inicializa os modelos base do ensemble."""
        self.modelos_base = {
            'rf': RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                         random_state=self.random_state),
            'gb': GradientBoostingClassifier(n_estimators=200,
                                             random_state=self.random_state),
            'lr': LogisticRegression(max_iter=1000,
                                     random_state=self.random_state),
            'nn': MLPClassifier(hidden_layer_sizes=(30, 15), alpha=0.1,
                                max_iter=1000, random_state=self.random_state,
                                early_stopping=True)
        }

        self.scalers = {nome: StandardScaler() for nome in self.modelos_base.keys()}
        self.modelos_treinados = {}
        self.pesos_modelos = {}

    def _inicializar_utilitarios(self):
        """Inicializa utilitários auxiliares."""
        self.db = DatabaseManager()
        self.feature_engineer = FeatureEngineer()
        self.features_selecionadas = []
        self.threshold_operacional = 0.5
        self.meta_modelo = None

    def _otimizar_hiperparametros_modelo(self, modelo, X: np.ndarray, y: np.ndarray):
        """Otimiza hiperparâmetros para um modelo específico."""
        from sklearn.model_selection import RandomizedSearchCV

        grades_parametros = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'gb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'lr': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2']
            },
            'nn': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }

        nome_modelo = self._identificar_tipo_modelo(modelo)

        if nome_modelo and nome_modelo in grades_parametros:
            busca = RandomizedSearchCV(
                modelo, grades_parametros[nome_modelo], n_iter=10, cv=3,
                scoring='accuracy', n_jobs=-1, random_state=self.random_state
            )
            busca.fit(X, y)
            return busca.best_estimator_

        return modelo

    def _identificar_tipo_modelo(self, modelo) -> Optional[str]:
        """Identifica o tipo do modelo para otimização."""
        for nome, modelo_base in self.modelos_base.items():
            if type(modelo) == type(modelo_base):
                return nome
        return None

    @staticmethod
    def _criar_divisoes_temporais(n_observacoes: int, n_splits: int = 4,
                                  purge_days: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Cria divisões temporais para validação walk-forward."""
        divisoes = []

        if n_splits <= 0:
            return divisoes

        tamanho_teste = max(int(n_observacoes / (n_splits + 1)), 1)

        for i in range(n_splits):
            fim_treino = tamanho_teste * (i + 1)
            inicio_teste = fim_treino + purge_days
            fim_teste = min(inicio_teste + tamanho_teste, n_observacoes)

            if inicio_teste >= fim_teste:
                continue

            indices_treino = np.arange(0, fim_treino)
            indices_teste = np.arange(inicio_teste, fim_teste)

            divisoes.append((indices_treino, indices_teste))

        return divisoes

    def selecionar_features_estaveis(self, X: pd.DataFrame, y: pd.Series,
                                     n_bootstraps: int = 100, proporcao_top: float = 0.2) -> List[str]:
        """Seleciona features usando método de estabilidade por bootstrap."""
        # Garantir colunas simples
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = X.columns.get_level_values(0)

        y = np.array(y).ravel().astype(int)

        # Importância por informação mútua
        scores_mi = mutual_info_classif(X, y, random_state=self.random_state)
        serie_mi = pd.Series(scores_mi, index=X.columns)
        top_mi = serie_mi.nlargest(15).index.tolist()

        # Sistema de votação
        votos = {coluna: 0 for coluna in X.columns}

        # Bonus para features com alta MI
        for feature in top_mi:
            votos[feature] += 10

        # Bootstrap com Random Forest
        n_manter = max(10, int(len(X.columns) * proporcao_top))
        rng = np.random.RandomState(self.random_state)

        for i in range(n_bootstraps):
            indices = rng.choice(len(X), size=int(len(X) * 0.8), replace=True)
            X_bootstrap = X.iloc[indices]
            y_bootstrap = y[indices]

            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                        random_state=self.random_state + i)
            rf.fit(X_bootstrap, y_bootstrap)

            # Top features por importância
            importancia = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = importancia.nlargest(n_manter).index

            for feature in top_features:
                votos[feature] += 1

        # Selecionar melhores features
        features_ordenadas = sorted(votos.items(), key=lambda x: x[1], reverse=True)
        selecionadas = [feature for feature, voto in features_ordenadas[:self.n_features]]

        print(f"🔍 Top 10 features selecionadas: {selecionadas[:10]}")
        print(f"📊 Scores das top 5: {[voto for _, voto in features_ordenadas[:5]]}")

        return selecionadas

    def treinar(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series,
                n_splits: int = 4, purge_days: int = 2) -> Dict[str, Any]:
        """Treina o sistema de classificação."""
        # Seleção de features
        self.features_selecionadas = self.selecionar_features_estaveis(X, y)
        X_selecionado = X[self.features_selecionadas].reset_index(drop=True)
        y = y.reset_index(drop=True)
        precos = precos.reset_index(drop=True)

        # Divisões temporais
        divisoes = self._criar_divisoes_temporais(len(X_selecionado), n_splits, purge_days)

        if not divisoes:
            raise ValueError("Divisões insuficientes para validação")

        # Treinamento dos modelos
        scores_cv, stds_cv = self._treinar_modelos_cv(X_selecionado, y, divisoes)
        self._ajustar_pesos_modelos(scores_cv, stds_cv)

        # Conjunto de holdout
        indices_treino, indices_teste = divisoes[-1]
        X_holdout = X_selecionado.iloc[indices_teste]
        y_holdout = y.iloc[indices_teste]
        precos_holdout = precos.iloc[indices_teste]

        # Calibrar threshold operacional
        self.calibrar_threshold_operacional(X_holdout, y_holdout)

        # Avaliar performance
        metricas = self._avaliar_performance(X_holdout, y_holdout, precos_holdout)

        # Treinar meta-modelo
        self._treinar_meta_modelo(X_holdout, y_holdout)

        # Salvar metadados do treinamento
        meta = {
            'features': self.features_selecionadas,
            'pesos': self.pesos_modelos,
            'cv_scores': scores_cv,
            'cv_stds': stds_cv,
            'threshold_operacional': float(self.threshold_operacional),
            'backtest': metricas
        }
        self.db.salvar_treino_metadata(meta)

        return metricas

    def _treinar_modelos_cv(self, X: pd.DataFrame, y: pd.Series,
                            divisoes: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Treina modelos com validação cruzada temporal."""
        scores_cv = {}
        stds_cv = {}

        for nome, modelo in self.modelos_base.items():
            scores_modelo = []

            for indices_treino, indices_teste in divisoes:
                X_treino = X.iloc[indices_treino]
                y_treino = y.iloc[indices_treino]
                X_teste = X.iloc[indices_teste]
                y_teste = y.iloc[indices_teste]

                # Escalar dados
                self.scalers[nome].fit(X_treino)
                X_treino_scaled = self.scalers[nome].transform(X_treino)
                X_teste_scaled = self.scalers[nome].transform(X_teste)

                # Otimizar hiperparâmetros se necessário
                if self.otimizar_hiperparametros:
                    modelo = self._otimizar_hiperparametros_modelo(modelo, X_treino_scaled, y_treino)

                # Treinar e avaliar
                modelo.fit(X_treino_scaled, y_treino)
                preds = modelo.predict(X_teste_scaled)
                score = accuracy_score(y_teste, preds)
                scores_modelo.append(score)

            # Calibrar modelo final
            X_scaled = self.scalers[nome].transform(X)
            modelo_calibrado = CalibratedClassifierCV(modelo, cv='prefit', method='isotonic')
            modelo_calibrado.fit(X_scaled, y)

            self.modelos_treinados[nome] = modelo_calibrado
            scores_cv[nome] = float(np.mean(scores_modelo))
            stds_cv[nome] = float(np.std(scores_modelo))

        return scores_cv, stds_cv

    def _ajustar_pesos_modelos(self, scores_cv: Dict[str, float], stds_cv: Dict[str, float]):
        """Ajusta pesos dos modelos baseados na performance CV."""
        for nome, score in scores_cv.items():
            # Penalizar alta variabilidade
            penalidade_std = max(0, 1 - (stds_cv[nome] / 0.1))
            self.pesos_modelos[nome] = score * penalidade_std

        # Normalizar pesos
        total = sum(self.pesos_modelos.values())
        if total > 0:
            for nome in self.pesos_modelos:
                self.pesos_modelos[nome] /= total
        else:
            # Distribuição uniforme se todos zero
            for nome in self.pesos_modelos:
                self.pesos_modelos[nome] = 1.0 / len(self.pesos_modelos)

    def calibrar_threshold_operacional(self, X: pd.DataFrame, y: pd.Series):
        """Calibra threshold operacional baseado em Brier Score."""
        probas = self.predict_proba(X)
        brier_scores = []
        thresholds = np.linspace(0.4, 0.6, 21)

        for threshold in thresholds:
            preds_binarias = (probas > threshold).astype(int)
            brier_scores.append(brier_score_loss(y, probas))

        melhor_threshold = thresholds[np.argmin(brier_scores)]
        self.threshold_operacional = melhor_threshold

        print(f"🎯 Threshold operacional calibrado: {melhor_threshold:.3f}")

    def _treinar_meta_modelo(self, X: pd.DataFrame, y: pd.Series):
        """Treina meta-modelo para combinar previsões."""
        probas_modelos = self._obter_probas_modelos(X)

        self.meta_modelo = LogisticRegression(random_state=self.random_state)
        self.meta_modelo.fit(probas_modelos, y)

    def _obter_probas_modelos(self, X: pd.DataFrame) -> np.ndarray:
        """Obtém probabilidades de todos os modelos."""
        probas_modelos = []

        for nome, modelo in self.modelos_treinados.items():
            X_scaled = self.scalers[nome].transform(X)
            proba = modelo.predict_proba(X_scaled)[:, 1]
            probas_modelos.append(proba)

        return np.column_stack(probas_modelos)

    def _avaliar_performance(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series) -> Dict[str, Any]:
        """Avalia performance do modelo no conjunto de holdout."""
        probas = self.predict_proba(X)
        preds = (probas > self.threshold_operacional).astype(int)

        # Métricas básicas
        acuracia = accuracy_score(y, preds)
        precisao = np.mean(preds == 1) if np.any(preds == 1) else 0

        # Backtest
        df_sinais = pd.DataFrame({
            'preco': precos.values,
            'proba': probas,
            'pred': preds
        })

        risk_analyzer = RiskAnalyzer()
        metricas_risco = risk_analyzer.backtest_sinais(df_sinais)

        return {
            'acuracia': acuracia,
            'precisao': precisao,
            'trades': metricas_risco['trades'],
            'retorno_total': metricas_risco['retorno_total'],
            'sharpe': metricas_risco['sharpe'],
            'max_drawdown': metricas_risco['max_drawdown'],
            'brier_score': brier_score_loss(y, probas)
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz probabilidades usando ensemble ponderado."""
        if not self.modelos_treinados:
            raise ValueError("Modelos não treinados. Execute treinar() primeiro.")

        X_selecionado = X[self.features_selecionadas]
        probas_modelos = self._obter_probas_modelos(X_selecionado)

        if self.meta_modelo:
            return self.meta_modelo.predict_proba(probas_modelos)[:, 1]
        else:
            # Combinação linear ponderada
            proba_final = np.zeros(len(X_selecionado))

            for nome, peso in self.pesos_modelos.items():
                X_scaled = self.scalers[nome].transform(X_selecionado)
                proba = self.modelos_treinados[nome].predict_proba(X_scaled)[:, 1]
                proba_final += peso * proba

            return proba_final

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz classes usando threshold operacional."""
        probas = self.predict_proba(X)
        return (probas > self.threshold_operacional).astype(int)

    def prever_direcao(self, X_novo: pd.DataFrame) -> Dict[str, Any]:
        """
        Faz previsão para novos dados.
        Retorna: probabilidade, predição e se deve operar.
        """
        if not hasattr(self, 'features_selecionadas') or not self.features_selecionadas:
            raise ValueError("Modelo não foi treinado ainda. Chame o método treinar() primeiro.")

        # Selecionar apenas as features usadas no treino
        X_selecionado = X_novo[self.features_selecionadas].reset_index(drop=True)

        # Calcular probabilidades do ensemble
        proba = np.array(self.predict_proba(X_selecionado)).ravel()

        # Fazer predição binária
        pred = (proba > self.threshold_operacional).astype(int).ravel()

        # Decidir se deve operar
        should_operate = bool((proba[-1] >= self.confidence_operar) and (pred[-1] == 1))

        resultado = {
            'probabilidade': float(proba[-1].item()),
            'predicao': int(pred[-1]),
            'should_operate': should_operate,
            'threshold_operacional': float(self.threshold_operacional),
            'limiar_confianca': float(self.confidence_operar)
        }

        # Salvar no banco de dados
        self.db.salvar_previsao({
            'probabilidade': float(proba[-1].item()),
            'predicao': int(pred[-1]),
            'metadados': {
                'should_operate': should_operate,
                'threshold_operacional': self.threshold_operacional,
                'limiar_confianca': self.confidence_operar
            }
        })

        return resultado

    def prever_e_gerar_sinais(self, X: pd.DataFrame, precos: pd.Series,
                              retornar_dataframe: bool = False) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        Gera sinais de trading baseados nas previsões do modelo.

        Args:
            X: DataFrame com features
            precos: Série com preços correspondentes
            retornar_dataframe: Se True, retorna DataFrame com sinais

        Returns:
            DataFrame com sinais ou tupla com predições e probabilidades
        """
        X_selecionado = X[self.features_selecionadas].reset_index(drop=True)
        proba = np.array(self.predict_proba(X_selecionado)).ravel()
        pred = (proba > self.threshold_operacional).astype(int).ravel()

        conf = proba
        high_conf = np.mean(conf >= self.confidence_operar)

        meta = {
            'ultimo_preco': float(precos.iloc[-1].item()) if len(precos) else None,
            'n_amostras': len(X_selecionado),
            'cobertura_alta_conf': float(high_conf),
            'threshold_operacional': float(self.threshold_operacional)
        }

        self.db.salvar_previsao({
            'predicao': int(pred[-1]) if len(pred) else None,
            'probabilidade': float(proba[-1].item()) if len(proba) else None,
            'metadados': meta
        })

        if retornar_dataframe:
            df = pd.DataFrame({
                'preco': precos.reset_index(drop=True).astype(float).to_numpy().ravel(),
                'proba': proba.ravel(),
                'pred': pred.ravel()
            })
            return df

        return pred, proba

    def operar(self, X: pd.DataFrame) -> Tuple[int, float]:
        """Decide se deve operar baseado na confiança da previsão."""
        proba = self.predict_proba(X)

        if len(proba) == 0:
            return 0, 0.0

        proba_final = proba[0]

        if abs(proba_final - 0.5) < (self.confidence_operar - 0.5):
            # Confiança insuficiente
            return 0, proba_final
        else:
            # Operar com direção baseada no threshold
            direcao = 1 if proba_final > self.threshold_operacional else -1
            return direcao, proba_final
