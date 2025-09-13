"""
🎯 MODELO DE CLASSIFICAÇÃO FINAL - FOCO EXCLUSIVO EM DIREÇÃO
Implementação definitiva para prever APENAS direção (Alta/Baixa)
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler


class ClassificacaoFinal:
    """
    Modelo FINAL focado EXCLUSIVAMENTE em classificação de direção
    """

    def __init__(self, ticker, test_size=0.20):
        self.ticker = ticker
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.is_trained = False
        self.test_size = test_size  # Permite configurar split treino/teste

        # Configurações otimizadas para estabilidade
        self.confidence_threshold = 0.75  # Threshold conservador
        self.n_features = 25  # Features balanceadas

        # Pesos equilibrados para estabilidade
        self.model_weights = {
            'rf': 0.27,  # Random Forest
            'gb': 0.26,  # Gradient Boosting
            'lr': 0.25,  # Logistic Regression
            'nn': 0.22  # Neural Network
        }

        # Métricas para análise
        self.ensemble_accuracy = 0
        self.baseline_accuracy = 0
        self.individual_accuracies = {}
        self.high_confidence_coverage = 0

        print(f"🎯 Modelo de Classificação inicializado para {ticker}")

    def _create_optimized_models(self):
        """Cria modelos otimizados para ALTA CONFIABILIDADE"""

        # Random Forest - Configuração para alta precisão
        self.models['rf'] = RandomForestClassifier(
            n_estimators=800,  # Mais árvores para estabilidade
            max_depth=15,  # Profundidade controlada
            min_samples_split=10,  # Mais conservador
            min_samples_leaf=5,  # Folhas maiores
            max_features=0.7,  # Mais features por árvore
            class_weight='balanced_subsample',  # Balanceamento por amostra
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True  # Out-of-bag score para validação
        )

        # Gradient Boosting - Configuração para alta confiança
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=500,  # Mais estimadores
            max_depth=8,  # Profundidade moderada
            learning_rate=0.02,  # Learning rate menor para estabilidade
            subsample=0.8,  # Subsample para regularização
            max_features=0.8,  # Mais features
            min_samples_split=15,  # Mais conservador
            min_samples_leaf=7,  # Folhas maiores
            random_state=42,
            validation_fraction=0.1,  # Validação interna
            n_iter_no_change=50  # Early stopping
        )

        # Logistic Regression - Regularização otimizada
        self.models['lr'] = LogisticRegression(
            C=0.1,  # Regularização moderada
            class_weight='balanced',
            solver='saga',  # Solver mais robusto
            penalty='elasticnet',  # Elastic net (L1 + L2)
            l1_ratio=0.5,  # Balanço entre L1 e L2
            random_state=42,
            max_iter=3000,
            tol=1e-6  # Tolerância menor para convergência
        )

        # Neural Network - Arquitetura otimizada para confiabilidade
        self.models['nn'] = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50, 25),  # Arquitetura mais profunda
            activation='tanh',  # Tanh para melhor gradiente
            solver='lbfgs',  # LBFGS para datasets pequenos
            alpha=0.01,  # Regularização mais forte
            learning_rate='constant',  # Learning rate constante
            learning_rate_init=0.001,
            max_iter=2000,  # Mais iterações
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,  # Mais dados para validação
            n_iter_no_change=100,  # Paciência maior
            tol=1e-6  # Tolerância menor
        )

        # Scalers específicos
        self.scalers['rf'] = RobustScaler()
        self.scalers['gb'] = RobustScaler()
        self.scalers['lr'] = StandardScaler()
        self.scalers['nn'] = StandardScaler()

    def obter_dados_d1(self):
        """Obtém dados até D-1 com 2 anos de histórico"""
        try:
            print(f"📊 Obtendo dados para {self.ticker} até D-1...")

            ontem = datetime.now() - timedelta(days=1)
            inicio = ontem - timedelta(days=730)

            print(f"📅 Período: {inicio.strftime('%Y-%m-%d')} a {ontem.strftime('%Y-%m-%d')}")

            stock = yf.Ticker(self.ticker)
            dados = stock.history(start=inicio, end=ontem + timedelta(days=1))

            if dados.empty:
                print("❌ Sem dados disponíveis")
                return None

            # Garante que não há dados de hoje
            hoje = datetime.now().date()
            dados = dados[dados.index.date < hoje]

            print(f"✅ {len(dados)} registros obtidos")
            print(f"📈 Último preço: ${dados['Close'].iloc[-1]:.2f} em {dados.index[-1].strftime('%Y-%m-%d')}")

            return dados

        except Exception as e:
            print(f"❌ Erro ao obter dados: {e}")
            return None

    def _create_classification_features(self, dados):
        """Cria features otimizadas para classificação de direção"""

        print("🔧 Criando features para classificação...")

        # Features básicas de retorno
        for periodo in [1, 2, 3, 5, 7, 10, 14, 21]:
            dados[f'retorno_{periodo}d'] = dados['Close'].pct_change(periodo)

        # Médias móveis e posições relativas
        for periodo in [5, 10, 20, 50, 100, 200]:
            dados[f'sma_{periodo}'] = dados['Close'].rolling(periodo).mean()
            dados[f'pos_sma_{periodo}'] = (dados['Close'] / dados[f'sma_{periodo}'] - 1) * 100

        # Razões de médias móveis (muito importantes para direção)
        dados['razao_sma_5_20'] = dados['sma_5'] / dados['sma_20']
        dados['razao_sma_10_50'] = dados['sma_10'] / dados['sma_50']
        dados['razao_sma_20_100'] = dados['sma_20'] / dados['sma_100']
        dados['razao_sma_50_200'] = dados['sma_50'] / dados['sma_200']

        # Momentum multi-período
        for periodo in [3, 7, 14, 21, 30]:
            dados[f'momentum_{periodo}d'] = (dados['Close'] / dados['Close'].shift(periodo) - 1) * 100

        # Volatilidade e regime de volatilidade
        for periodo in [5, 10, 20, 50]:
            dados[f'vol_{periodo}d'] = dados['retorno_1d'].rolling(periodo).std() * 100
            dados[f'vol_regime_{periodo}'] = dados[f'vol_{periodo}d'] / dados[f'vol_{periodo}d'].rolling(periodo).mean()

        # RSI multi-período
        for periodo in [7, 14, 21, 30]:
            delta = dados['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(periodo).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
            rs = gain / loss
            dados[f'rsi_{periodo}'] = 100 - (100 / (1 + rs))
            dados[f'rsi_{periodo}_overbought'] = (dados[f'rsi_{periodo}'] > 70).astype(int)
            dados[f'rsi_{periodo}_oversold'] = (dados[f'rsi_{periodo}'] < 30).astype(int)

        # MACD e derivados
        ema_12 = dados['Close'].ewm(span=12).mean()
        ema_26 = dados['Close'].ewm(span=26).mean()
        dados['macd'] = ema_12 - ema_26
        dados['macd_signal'] = dados['macd'].ewm(span=9).mean()
        dados['macd_histogram'] = dados['macd'] - dados['macd_signal']
        dados['macd_bullish'] = (dados['macd'] > dados['macd_signal']).astype(int)
        dados['macd_crossover'] = ((dados['macd'] > dados['macd_signal']) &
                                   (dados['macd'].shift(1) <= dados['macd_signal'].shift(1))).astype(int)

        # Volume e análise de volume
        for periodo in [5, 10, 20, 50]:
            dados[f'volume_sma_{periodo}'] = dados['Volume'].rolling(periodo).mean()
            dados[f'volume_ratio_{periodo}'] = dados['Volume'] / dados[f'volume_sma_{periodo}']

        dados['volume_price_trend'] = dados['Volume'] * dados['retorno_1d']
        dados['volume_breakout'] = (dados['volume_ratio_20'] > 2.0).astype(int)

        # Suporte e resistência
        for periodo in [10, 20, 50]:
            dados[f'high_{periodo}d'] = dados['High'].rolling(periodo).max()
            dados[f'low_{periodo}d'] = dados['Low'].rolling(periodo).min()
            dados[f'price_position_{periodo}'] = ((dados['Close'] - dados[f'low_{periodo}d']) /
                                                  (dados[f'high_{periodo}d'] - dados[f'low_{periodo}d']))
            dados[f'near_high_{periodo}'] = (dados[f'price_position_{periodo}'] > 0.8).astype(int)
            dados[f'near_low_{periodo}'] = (dados[f'price_position_{periodo}'] < 0.2).astype(int)

        # Padrões de velas
        dados['body_size'] = abs(dados['Close'] - dados['Open']) / (dados['High'] - dados['Low'])
        dados['upper_shadow'] = (dados['High'] - np.maximum(dados['Open'], dados['Close'])) / (
                dados['High'] - dados['Low'])
        dados['lower_shadow'] = (np.minimum(dados['Open'], dados['Close']) - dados['Low']) / (
                dados['High'] - dados['Low'])
        dados['doji'] = (dados['body_size'] < 0.1).astype(int)
        dados['hammer'] = ((dados['lower_shadow'] > 0.6) & (dados['upper_shadow'] < 0.1)).astype(int)

        # Tendências de múltiplos períodos
        for periodo in [3, 5, 10, 20, 50]:
            dados[f'trend_{periodo}d'] = (dados['Close'] > dados['Close'].shift(periodo)).astype(int)
            dados[f'trend_strength_{periodo}'] = (dados['Close'] / dados['Close'].shift(periodo) - 1) * 100

        # Features de aceleração
        dados['price_acceleration'] = dados['retorno_1d'] - dados['retorno_1d'].shift(1)
        dados['volume_acceleration'] = dados['volume_ratio_5'] - dados['volume_ratio_5'].shift(1)

        print(
            f"✅ Features criadas: {len([col for col in dados.columns if dados[col].dtype in ['float64', 'int64']])} numéricas")

        return dados

    def treinar(self, verbose=True):
        """Treina o modelo de classificação"""

        if verbose:
            print(f"\n🎯 TREINAMENTO CLASSIFICAÇÃO FINAL - {self.ticker}")
            print("=" * 70)

        try:
            # Obtém dados
            dados = self.obter_dados_d1()
            if dados is None:
                return False

            # Cria features
            dados_features = self._create_classification_features(dados)

            # Prepara dados
            X, y = self._preparar_dados_classificacao(dados_features)

            if len(X) < 200:
                print("❌ Dados insuficientes para treinamento")
                return False

            print(f"📊 Dados preparados: {len(X)} amostras, {X.shape[1]} features")

            # Divide dados usando test_size configurável
            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Armazena para análise
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test

            print(f"📈 Divisão: {len(X_train)} treino, {len(X_test)} teste")
            print(f"📊 Distribuição treino: {y_train.mean():.1%} alta, {1 - y_train.mean():.1%} baixa")
            print(f"📊 Distribuição teste: {y_test.mean():.1%} alta, {1 - y_test.mean():.1%} baixa")

            # Cria modelos
            self._create_optimized_models()

            # Seleção de features ROBUSTA para alta confiabilidade
            print(f"🔍 Selecionando top {self.n_features} features mais estáveis...")

            # Salva features de treinamento para usar na previsão
            self.training_features = X_train.columns.tolist()

            X_train_selected, X_test_selected = self._select_robust_features(X_train, X_test, y_train)

            # Treina cada modelo
            model_scores = {}
            for model_name in self.models.keys():
                score = self._train_individual_model(model_name, X_train_selected, y_train)
                model_scores[model_name] = score

            # Ajusta pesos baseado na performance
            self._adjust_weights(model_scores)

            # Avalia ensemble
            ensemble_score = self._evaluate_ensemble(X_test_selected, y_test)

            # Análise de confiança
            confidence_analysis = self._analyze_confidence(X_test_selected, y_test)

            # Métricas finais
            baseline = max(y_test.mean(), 1 - y_test.mean())
            melhoria = ensemble_score - baseline

            print(f"\n📊 RESULTADOS FINAIS:")
            print(f"   🎯 Ensemble: {ensemble_score:.3f}")
            print(f"   📊 Baseline: {baseline:.3f}")
            print(f"   🚀 Melhoria: {melhoria:+.3f}")

            for model_name, score in model_scores.items():
                weight = self.model_weights[model_name]
                print(f"   🤖 {model_name.upper()}: {score:.3f} (peso: {weight:.2f})")

            print(f"   🎯 Alta confiança: {confidence_analysis['high_conf_accuracy']:.3f} "
                  f"({confidence_analysis['high_conf_coverage']:.1%} cobertura)")

            # Features mais importantes (usando Random Forest como referência)
            rf_model = self.models['rf']
            feature_importance = pd.Series(rf_model.feature_importances_, index=X_train_selected.columns)
            top_features = feature_importance.nlargest(10)

            print(f"\n🔍 TOP 10 FEATURES MAIS IMPORTANTES:")
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                print(f"   {i:2d}. {feature} ({importance:.4f})")

            self.is_trained = True
            print("✅ Treinamento concluído com sucesso!")

            return True

        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _preparar_dados_classificacao(self, dados):
        """Prepara dados especificamente para classificação"""

        # Seleciona apenas features numéricas
        feature_cols = [col for col in dados.columns
                        if col not in ['Date', 'ticker', 'data']
                        and dados[col].dtype in ['float64', 'int64']]

        X = dados[feature_cols].fillna(0)

        # Target: direção do próximo dia (APENAS classificação)
        y = (dados['Close'].shift(-1) > dados['Close']).astype(int)

        # Remove última linha (sem target futuro)
        X = X.iloc[:-1]
        y = y.iloc[:-1]

        # Remove NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        return X, y

    def _train_individual_model(self, model_name, X_train, y_train):
        """Treina modelo individual"""

        print(f"🤖 Treinando {model_name.upper()}...")

        # Aplica scaler
        X_scaled = self.scalers[model_name].fit_transform(X_train)

        # Treina modelo
        self.models[model_name].fit(X_scaled, y_train)

        # Validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(
            self.models[model_name], X_scaled, y_train,
            cv=tscv, scoring='accuracy'
        )

        avg_score = cv_scores.mean()
        print(f"   CV Accuracy: {avg_score:.3f} ± {cv_scores.std():.3f}")

        return avg_score

    def _adjust_weights(self, model_scores):
        """Ajusta pesos baseado na performance"""

        total_score = sum(model_scores.values())
        if total_score > 0:
            for model_name in self.model_weights.keys():
                if model_name in model_scores:
                    performance_weight = model_scores[model_name] / total_score
                    original_weight = self.model_weights[model_name]
                    self.model_weights[model_name] = 0.8 * performance_weight + 0.2 * original_weight

        # Normaliza pesos
        total_weight = sum(self.model_weights.values())
        for model_name in self.model_weights.keys():
            self.model_weights[model_name] /= total_weight

    def _evaluate_ensemble(self, X_test, y_test):
        """Avalia ensemble"""

        ensemble_proba = np.zeros(len(X_test))

        for model_name in self.models.keys():
            X_scaled = self.scalers[model_name].transform(X_test)
            proba = self.models[model_name].predict_proba(X_scaled)[:, 1]
            ensemble_proba += proba * self.model_weights[model_name]

        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        return accuracy_score(y_test, ensemble_pred)

    def _analyze_confidence(self, X_test, y_test):
        """Analisa confiança das previsões com método melhorado"""

        ensemble_proba = np.zeros(len(X_test))
        individual_probas = {}

        for model_name in self.models.keys():
            X_scaled = self.scalers[model_name].transform(X_test)
            proba = self.models[model_name].predict_proba(X_scaled)[:, 1]
            individual_probas[model_name] = proba
            ensemble_proba += proba * self.model_weights[model_name]

        # NOVO: Confiança baseada em consenso entre modelos + distância de 0.5
        confidence = self._calculate_improved_confidence(individual_probas, ensemble_proba)

        high_conf_mask = confidence > self.confidence_threshold

        if high_conf_mask.sum() > 0:
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            high_conf_accuracy = accuracy_score(y_test[high_conf_mask], ensemble_pred[high_conf_mask])
            high_conf_coverage = high_conf_mask.mean()
        else:
            high_conf_accuracy = 0
            high_conf_coverage = 0

        return {
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_coverage': high_conf_coverage,
            'avg_confidence': confidence.mean()
        }

    def _calculate_improved_confidence(self, individual_probas, ensemble_proba):
        """Calcula confiança melhorada baseada em consenso e certeza"""

        # Converte probabilidades em decisões (0 ou 1)
        decisions = {}
        for model_name, probas in individual_probas.items():
            decisions[model_name] = (probas > 0.5).astype(int)

        # Calcula consenso (quantos modelos concordam)
        decision_matrix = np.array(list(decisions.values()))
        consensus = np.abs(decision_matrix.mean(axis=0) - 0.5) * 2  # 0 a 1

        # Calcula certeza (quão longe de 0.5 está o ensemble)
        certainty = np.abs(ensemble_proba - 0.5) * 2  # 0 a 1

        # Combina consenso e certeza (dando mais peso ao consenso)
        confidence = 0.7 * consensus + 0.3 * certainty

        return confidence

    def _select_robust_features(self, X_train, X_test, y_train):
        """Seleciona features mais robustas combinando múltiplos métodos"""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier

        # Método 1: Mutual Information
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=self.n_features * 2)
        X_mi = selector_mi.fit_transform(X_train, y_train)
        features_mi = set(X_train.columns[selector_mi.get_support()])

        # Método 2: F-score
        selector_f = SelectKBest(score_func=f_classif, k=self.n_features * 2)
        X_f = selector_f.fit_transform(X_train, y_train)
        features_f = set(X_train.columns[selector_f.get_support()])

        # Método 3: Random Forest Feature Importance
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train, y_train)
        feature_importance = pd.Series(rf_temp.feature_importances_, index=X_train.columns)
        top_rf_features = set(feature_importance.nlargest(self.n_features * 2).index)

        # Combina métodos: features que aparecem em pelo menos 2 dos 3 métodos
        feature_votes = {}
        for feature in X_train.columns:
            votes = 0
            if feature in features_mi: votes += 1
            if feature in features_f: votes += 1
            if feature in top_rf_features: votes += 1
            feature_votes[feature] = votes

        # Seleciona features com mais votos
        selected_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        final_features = [f[0] for f in selected_features[:self.n_features]]

        # Salva features selecionadas e cria mapeamento
        self.selected_features = final_features
        self.feature_mapping = {i: feature for i, feature in enumerate(final_features)}

        # Não usa mais SelectKBest, apenas retorna as features selecionadas
        return X_train[final_features], X_test[final_features]

    def _proximo_dia_util(self):
        """Calcula o próximo dia útil (segunda a sexta)"""
        from datetime import datetime, timedelta

        hoje = datetime.now()
        proximo_dia = hoje + timedelta(days=1)

        # Se for sábado (5) ou domingo (6), pula para segunda
        while proximo_dia.weekday() >= 5:  # 5=sábado, 6=domingo
            proximo_dia += timedelta(days=1)

        return proximo_dia

    def _deve_fazer_previsao(self):
        """Verifica se deve fazer previsão baseado no dia da semana"""
        from datetime import datetime

        hoje = datetime.now()
        dia_semana = hoje.weekday()  # 0=segunda, 6=domingo

        # Se for sábado ou domingo, mostrar previsão para segunda
        if dia_semana >= 5:
            return True, "Final de semana - Previsão para segunda-feira"

        # Se for sexta, avisar que previsão é para segunda
        if dia_semana == 4:  # sexta
            return True, "Previsão para segunda-feira (próximo dia útil)"

        return True, "Previsão para próximo dia útil"

    def prever_direcao(self):
        """Faz previsão de direção para o próximo dia útil"""

        if not self.is_trained:
            print("❌ Modelo não foi treinado ainda!")
            return None

        # Verifica qual dia será a previsão
        deve_prever, motivo = self._deve_fazer_previsao()

        try:
            print(f"\n🔮 PREVISÃO PARA {self.ticker} - PRÓXIMO DIA ÚTIL")
            print("-" * 50)

            if "segunda-feira" in motivo.lower():
                print(f"📅 {motivo}")
                print()

            # Obtém dados atualizados
            dados = self.obter_dados_d1()
            if dados is None:
                return None

            # Cria features
            dados_features = self._create_classification_features(dados)

            # Prepara dados (só features, sem target)
            feature_cols = [col for col in dados_features.columns
                            if col not in ['Date', 'ticker', 'data']
                            and dados_features[col].dtype in ['float64', 'int64']]

            X = dados_features[feature_cols].fillna(0)

            # Aplica seleção de features (garantindo mesmas features do treinamento)
            if hasattr(self, 'selected_features'):
                # Usa apenas as features que foram selecionadas no treinamento
                # Adiciona features faltantes com valor 0
                for feature in self.selected_features:
                    if feature not in X.columns:
                        X[feature] = 0

                # Seleciona apenas as features treinadas
                X_selected = X[self.selected_features]
            else:
                # Fallback: usa todas as features
                X_selected = X

            # Predições ensemble
            ensemble_proba = 0
            individual_probas = {}

            for model_name in self.models.keys():
                X_scaled = self.scalers[model_name].transform(X_selected)
                proba = self.models[model_name].predict_proba(X_scaled)[-1, 1]

                individual_probas[model_name] = proba
                ensemble_proba += proba * self.model_weights[model_name]

            # Decisão final
            prediction = int(ensemble_proba > 0.5)

            # NOVO: Usa cálculo melhorado de confiança
            confidence = self._calculate_improved_confidence(
                {k: np.array([v]) for k, v in individual_probas.items()},
                np.array([ensemble_proba])
            )[0]

            should_trade = confidence > self.confidence_threshold

            # Data da previsão (próximo dia útil)
            proximo_dia_util = self._proximo_dia_util()
            data_previsao = proximo_dia_util.strftime('%Y-%m-%d')
            dia_semana_previsao = proximo_dia_util.strftime('%A')

            # Traduz dia da semana
            dias_pt = {
                'Monday': 'Segunda-feira',
                'Tuesday': 'Terça-feira',
                'Wednesday': 'Quarta-feira',
                'Thursday': 'Quinta-feira',
                'Friday': 'Sexta-feira'
            }
            dia_pt = dias_pt.get(dia_semana_previsao, dia_semana_previsao)

            # Resultado
            resultado = {
                'ticker': self.ticker,
                'prediction': prediction,
                'direction': 'ALTA' if prediction == 1 else 'BAIXA',
                'probability': ensemble_proba,
                'confidence': confidence,
                'should_trade': should_trade,
                'individual_probas': individual_probas,
                'last_price': dados['Close'].iloc[-1],
                'data_ultima': dados.index[-1].strftime('%Y-%m-%d'),
                'data_previsao': data_previsao,
                'dia_previsao': dia_pt
            }

            print(f"📊 Dados até: {resultado['data_ultima']}")
            print(f"💰 Último preço: ${resultado['last_price']:.2f}")
            print(f"📅 Previsão para: {resultado['data_previsao']} ({resultado['dia_previsao']})")
            print(f"🎯 Previsão: {resultado['direction']}")
            print(f"📊 Probabilidade: {resultado['probability']:.3f}")
            print(f"🎯 Confiança: {resultado['confidence']:.3f}")
            print(f"💡 Deve operar: {'✅ SIM' if resultado['should_trade'] else '❌ NÃO'}")

            print(f"\n🤖 Detalhes por modelo:")
            for model_name, proba in individual_probas.items():
                weight = self.model_weights[model_name]
                direction = 'ALTA' if proba > 0.5 else 'BAIXA'
                print(f"   {model_name.upper()}: {direction} ({proba:.3f}) - peso: {weight:.2f}")

            return resultado

        except Exception as e:
            print(f"❌ Erro na previsão: {e}")
            return None
