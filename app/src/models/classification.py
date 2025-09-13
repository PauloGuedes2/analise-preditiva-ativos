# classificacao_final_refinado.py
"""
Versão refinada do pipeline de classificação (PT-BR)
- Expanding walk-forward com purge
- Calibração de probabilidades
- Stability selection por bootstrap
- Threshold operacional derivado da validação
- Correções para warnings do sklearn e pandas
"""

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.data.database_manager import DatabaseManagerRefinado
from src.models.feature_engineer import FeatureEngineerRefinado
from src.utils.risk_analyzer import RiskAnalyzerRefinado


class ClassificacaoFinalRefinado:
    def __init__(self, n_features: int = 25, random_state: int = 42,
                 confidence_operar: float = 0.60, otimizar_hiperparametros: bool = True):

        self.n_features = n_features
        self.random_state = random_state
        self.confidence_operar = confidence_operar
        self.otimizar_hiperparametros = otimizar_hiperparametros

        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=self.random_state),
            'gb': GradientBoostingClassifier(n_estimators=200, random_state=self.random_state),
            'lr': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'nn': MLPClassifier(hidden_layer_sizes=(30, 15), alpha=0.1, max_iter=1000,
                                random_state=self.random_state, early_stopping=True)
        }

        self.scalers = {k: StandardScaler() for k in self.base_models.keys()}

        self.modelos_treinados: Dict[str, Any] = {}
        self.pesos_modelos: Dict[str, float] = {}
        self.features_selecionadas: List[str] = []
        self.threshold_operacional: float = 0.5
        self.db = DatabaseManagerRefinado()
        self.feature_engineer = FeatureEngineerRefinado()

    def _otimizar_hiperparametros_modelo(self, modelo, X, y):
        """Otimiza hiperparâmetros para cada modelo"""
        from sklearn.model_selection import RandomizedSearchCV

        param_grids = {
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

        modelo_nome = None
        for nome, mod in self.base_models.items():
            if type(modelo) == type(mod):
                modelo_nome = nome
                break

        if modelo_nome and modelo_nome in param_grids:
            search = RandomizedSearchCV(
                modelo, param_grids[modelo_nome], n_iter=10, cv=3,
                scoring='accuracy', n_jobs=-1, random_state=self.random_state
            )
            search.fit(X, y)
            return search.best_estimator_

        return modelo

    # ----------------------------
    def _divisoes_expansivas(self, n_splits: int, purge_days: int, n_observacoes: int) -> List[
        Tuple[np.ndarray, np.ndarray]]:
        splits = []
        if n_splits <= 0:
            return splits
        test_size = max(int(n_observacoes / (n_splits + 1)), 1)
        for i in range(n_splits):
            train_end = test_size * (i + 1)
            test_start = train_end + purge_days
            test_end = min(test_start + test_size, n_observacoes)
            if test_start >= test_end:
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
        return splits

    # ----------------------------
    def selecao_estavel(self, X: pd.DataFrame, y: pd.Series, n_boot: int = 100, proporcao_top: float = 0.2):
        # Garantir que X tenha colunas simples
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = X.columns.get_level_values(0)

        y = np.array(y).ravel()
        y = y.astype(int)

        # Usar importância mutual + random forest
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier

        # Calcular mutual information
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_series = pd.Series(mi_scores, index=X.columns)

        # Top features por MI
        top_mi = mi_series.nlargest(15).index.tolist()

        votes = {c: 0 for c in X.columns}

        # Dar peso extra para features com alta MI
        for feature in top_mi:
            votes[feature] += 10

        # Bootstrap com Random Forest
        n_keep = max(10, int(len(X.columns) * proporcao_top))
        rng = np.random.RandomState(self.random_state)

        for i in range(n_boot):
            idx = rng.choice(len(X), size=int(len(X) * 0.8), replace=True)
            Xb = X.iloc[idx]
            yb = y[idx]

            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                        random_state=self.random_state + i)
            rf.fit(Xb, yb)

            # Top features por importância
            imp = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = imp.nlargest(n_keep).index

            for f in top_features:
                votes[f] += 1

        # Selecionar as melhores features
        voted = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        selected = [k for k, v in voted[:self.n_features]]

        print(f"🔍 Top 10 features selecionadas: {selected[:10]}")
        print(f"📊 Scores das top 5: {[v for k, v in voted[:5]]}")

        self.features_selecionadas = selected
        return selected

    # ----------------------------
    def treinar(self, X: pd.DataFrame, y: pd.Series, precos: pd.Series, n_splits: int = 4, purge_days: int = 1):

        sel = self.selecao_estavel(X, y)
        Xs = X[sel].reset_index(drop=True)
        y = y.reset_index(drop=True)
        precos = precos.reset_index(drop=True)

        splits = self._divisoes_expansivas(n_splits=n_splits, purge_days=purge_days,
                                           n_observacoes=len(Xs))
        if len(splits) == 0:
            raise ValueError("Divisões insuficientes")

        X_full = X.copy()
        sel = self.selecao_estavel(X_full, y)

        Xs = X_full[sel].reset_index(drop=True)
        y = y.reset_index(drop=True)

        precos = precos.reset_index(drop=True)

        n_obs = len(Xs)
        splits = self._divisoes_expansivas(n_splits=n_splits, purge_days=purge_days, n_observacoes=n_obs)
        if len(splits) == 0:
            raise ValueError("Número de observações insuficiente para as divisões expansivas solicitadas.")

        cv_scores = {}
        cv_stds = {}
        for nome, modelo in self.base_models.items():
            scores = []
            for train_idx, test_idx in splits:
                Xtr = Xs.iloc[train_idx]
                ytr = y.iloc[train_idx]
                Xte = Xs.iloc[test_idx]
                yte = y.iloc[test_idx]

                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)

                ytr = np.array(ytr).ravel()
                yte = np.array(yte).ravel()

                m = modelo.__class__(**modelo.get_params())
                if hasattr(m, 'random_state'):
                    try:
                        m.set_params(random_state=self.random_state)
                    except Exception:
                        pass
                m.fit(Xtr_s, ytr)
                preds = m.predict(Xte_s)
                scores.append(accuracy_score(yte, preds))

            if self.otimizar_hiperparametros:
                m = self._otimizar_hiperparametros_modelo(m, Xtr_s, ytr)

            cv_scores[nome] = float(np.mean(scores)) if scores else 0.0
            cv_stds[nome] = float(np.std(scores)) if scores else 0.0

            scaler_full = StandardScaler().fit(Xs)
            Xs_s = scaler_full.transform(Xs)
            y_full = np.array(y).ravel()
            modelo.fit(Xs_s, y_full)

            n_cal = max(50, int(len(Xs_s) * 0.15))  # Mais dados para calibração
            if n_cal >= 50:  # Mínimo de 50 amostras
                X_fit, X_cal = Xs_s[:-n_cal], Xs_s[-n_cal:]
                y_fit, y_cal = y_full[:-n_cal], y_full[-n_cal:]

                modelo.fit(X_fit, y_fit)

                # Testar ambos os métodos de calibração
                try:
                    calibrated_sigmoid = CalibratedClassifierCV(modelo, method='sigmoid', cv='prefit')
                    calibrated_sigmoid.fit(X_cal, y_cal)

                    calibrated_isotonic = CalibratedClassifierCV(modelo, method='isotonic', cv='prefit')
                    calibrated_isotonic.fit(X_cal, y_cal)

                    # Escolher o melhor método
                    proba_sigmoid = calibrated_sigmoid.predict_proba(X_cal)[:, 1]
                    proba_isotonic = calibrated_isotonic.predict_proba(X_cal)[:, 1]

                    score_sigmoid = brier_score_loss(y_cal, proba_sigmoid)
                    score_isotonic = brier_score_loss(y_cal, proba_isotonic)

                    self.modelos_treinados[
                        nome] = calibrated_sigmoid if score_sigmoid < score_isotonic else calibrated_isotonic
                except Exception as e:
                    print(f"⚠️ Calibração falhou para {nome}: {e}")
                    self.modelos_treinados[nome] = modelo
            else:
                self.modelos_treinados[nome] = modelo

            self.scalers[nome] = scaler_full

        self._ajustar_pesos_por_cv(cv_scores, cv_stds)

        ultimo_train_idx, ultimo_test_idx = splits[-1]
        X_hold = Xs.iloc[ultimo_test_idx]
        y_hold = y.iloc[ultimo_test_idx]
        precos_hold = precos.iloc[ultimo_test_idx]

        self.calcular_threshold_operacional(X_hold, y_hold)

        probas_ensemble = self._proba_ensemble(X_hold)
        print(f"📊 Estatísticas das probabilidades:")
        print(f"   Média: {probas_ensemble.mean():.3f}")
        print(f"   Máxima: {probas_ensemble.max():.3f}")
        print(f"   Mínima: {probas_ensemble.min():.3f}")
        print(f"   % acima de 0.75: {(probas_ensemble > 0.75).mean():.1%}")

        ra = RiskAnalyzerRefinado()
        df_signals = self.prever_e_gerar_sinais(X_hold, precos_hold, retornar_dataframe=True)
        backtest_metrics = ra.backtest_sinais(df_signals, custo_por_trade_pct=0.0005)

        meta = {
            'features': self.features_selecionadas,
            'pesos': self.pesos_modelos,
            'cv_scores': cv_scores,
            'cv_stds': cv_stds,
            'threshold_operacional': float(self.threshold_operacional),
            'backtest': backtest_metrics
        }
        self.db.salvar_treino_metadata(meta)

        probas_validacao = self._proba_ensemble(X_hold)
        self.confidence_operar = max(self.calcular_limiar_confianca_auto(probas_validacao), 0.55)
        print(f"🎯 Limiar de confiança automático: {self.confidence_operar:.3f}")

        print("🧠 Treinando meta-modelo...")

        try:
            probas_base = {}
            for nome, modelo in self.modelos_treinados.items():
                scaler = self.scalers[nome]
                Xs_scaled = scaler.transform(X_hold)
                probas_base[nome] = modelo.predict_proba(Xs_scaled)[:, 1]

            # Treinar meta-modelo (Stacking)
            X_meta = np.column_stack(list(probas_base.values()))
            meta_model = LogisticRegression(random_state=self.random_state)
            meta_model.fit(X_meta, y_hold)

            self.meta_model = meta_model
            print("✅ Meta-modelo treinado")
        except Exception as e:
            print(f"⚠️  Meta-modelo não pôde ser treinado: {e}")
            self.meta_model = None

        return meta

    def _treinar_meta_modelo(self, X, y, previsoes_base):
        from sklearn.linear_model import LogisticRegression

        # Combinar previsões dos modelos base
        X_meta = np.column_stack([previsoes_base[modelo] for modelo in previsoes_base])

        meta_model = LogisticRegression()
        meta_model.fit(X_meta, y)
        return meta_model

    def _ajustar_pesos_por_cv(self, cv_scores: Dict[str, float], cv_stds: Dict[str, float]):
        nomes = list(self.base_models.keys())
        scores = np.array([cv_scores.get(n, 0.0) for n in nomes], dtype=float)
        stds = np.array([cv_stds.get(n, 1.0) for n in nomes], dtype=float)
        penalized = scores - 0.5 * stds
        penalized = np.clip(penalized, a_min=0.0001, a_max=None)
        pesos = penalized / np.sum(penalized)
        self.pesos_modelos = {n: float(p) for n, p in zip(nomes, pesos)}
        return self.pesos_modelos

    def calcular_threshold_operacional(self, X_hold: pd.DataFrame, y_hold: pd.Series, inicio: float = 0.45,
                                       fim: float = 0.9, passos: int = 46):
        if len(X_hold) == 0:
            self.threshold_operacional = 0.5
            return self.threshold_operacional
        probas_ensemble = self._proba_ensemble(X_hold)
        best_thr = 0.5
        best_metric = -999
        for thr in np.linspace(inicio, fim, passos):
            preds = (probas_ensemble > thr).astype(int)
            acc = accuracy_score(np.array(y_hold).ravel(), preds)
            if acc > best_metric:
                best_metric = acc
                best_thr = thr
        self.threshold_operacional = float(best_thr)
        return self.threshold_operacional

    def _proba_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        # Previsões dos modelos base
        probas_base = {}
        for nome, modelo in self.modelos_treinados.items():
            scaler = self.scalers.get(nome, StandardScaler())
            Xs = scaler.transform(X)
            try:
                p = modelo.predict_proba(Xs)[:, 1]
            except Exception:
                try:
                    raw = modelo.decision_function(Xs)
                    p = 1 / (1 + np.exp(-raw))
                except Exception:
                    p = np.zeros(len(Xs))
            probas_base[nome] = p

        # Usar meta-modelo se disponível
        if hasattr(self, 'meta_model'):
            X_meta = np.column_stack(list(probas_base.values()))
            return self.meta_model.predict_proba(X_meta)[:, 1]
        else:
            # Fallback para média ponderada
            probas = []
            for nome, p in probas_base.items():
                peso = self.pesos_modelos.get(nome, 0.25)
                probas.append(p * peso)
            return np.sum(np.vstack(probas), axis=0)

    def prever_e_gerar_sinais(self, X: pd.DataFrame, precos: pd.Series, retornar_dataframe: bool = False):
        Xs = X[self.features_selecionadas].reset_index(drop=True)
        proba = np.array(self._proba_ensemble(Xs)).ravel()
        pred = (proba > self.threshold_operacional).astype(int).ravel()

        conf = proba
        high_conf = np.mean(conf >= self.confidence_operar)
        meta = {
            'ultimo_preco': float(precos.iloc[-1].item()) if len(precos) else None,
            'n_amostras': len(Xs),
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

    def calcular_limiar_confianca_auto(self, probas_validacao, y_true=None):
        """Limiar dinâmico baseado na qualidade das previsões"""
        base = np.percentile(probas_validacao, 65)  # Percentil 65

        # Se tivermos labels verdadeiras, ajustar baseado na acurácia
        if y_true is not None:
            acc = accuracy_score(y_true, (probas_validacao > 0.5).astype(int))
            if acc > 0.55:  # Se a acurácia for boa, podemos ser menos conservadores
                base = np.percentile(probas_validacao, 60)
            elif acc < 0.52:  # Se a acurácia for baixa, ser mais conservador
                base = np.percentile(probas_validacao, 70)

        # Limites mínimos e máximos
        base = max(base, 0.55)  # Mínimo de 55%
        base = min(base, 0.70)  # Máximo de 70%

        return base

    def _treinar_meta_modelo_avancado(self, X_hold, y_hold):
        """Meta-modelo mais sofisticado"""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier

        # Coletar previsões de todos os modelos
        probas_base = {}
        for nome, modelo in self.modelos_treinados.items():
            scaler = self.scalers[nome]
            Xs_scaled = scaler.transform(X_hold)
            probas_base[nome] = modelo.predict_proba(Xs_scaled)[:, 1]

        # Criar dataset meta
        X_meta = np.column_stack(list(probas_base.values()))

        # Adicionar features originais selecionadas
        X_meta = np.hstack([X_meta, X_hold[self.features_selecionadas[:5]].values])

        # Testar diferentes meta-modelos
        models = {
            'logistic': LogisticRegression(random_state=self.random_state),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'nn': MLPClassifier(hidden_layer_sizes=(10, 5), random_state=self.random_state)
        }

        best_score = -1
        best_model = None

        for name, model in models.items():
            try:
                model.fit(X_meta, y_hold)
                score = model.score(X_meta, y_hold)
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                continue

        return best_model

    def prever_direcao(self, X_novo: pd.DataFrame) -> Dict[str, Any]:
        """
        Faz previsão para novos dados
        Retorna: probabilidade, predição e se deve operar
        """
        if not hasattr(self, 'features_selecionadas') or not self.features_selecionadas:
            raise ValueError("Modelo não foi treinado ainda. Chame o método treinar() primeiro.")

        # Selecionar apenas as features usadas no treino
        Xs = X_novo[self.features_selecionadas].reset_index(drop=True)

        # Calcular probabilidades do ensemble
        proba = np.array(self._proba_ensemble(Xs)).ravel()

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
