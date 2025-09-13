import numpy as np
import pandas as pd


class EngenheiriaFeatures:
    """
    Classe responsável por criar features técnicas para análise de ações.
    
    Esta classe implementa diversos indicadores técnicos e features de mercado
    para alimentar modelos de machine learning, garantindo que não haja vazamento
    de dados futuros (data leakage).
    """
    
    def __init__(self):
        """
        Inicializa o engenheiro de features.
        
        Define o período mínimo necessário para calcular features confiáveis.
        """
        self.periodos_minimos = 20  # Período mínimo para features confiáveis
        
    def criar_features(self, df):
        """
        Cria features técnicas APENAS com dados históricos - SEM vazamento de dados futuros.
        
        Args:
            df (pandas.DataFrame): DataFrame com dados OHLCV da ação
            
        Returns:
            pandas.DataFrame: DataFrame com features técnicas adicionadas
        """
        df = df.copy()
        
        # Achata colunas multi-nível se necessário (yfinance às vezes retorna multi-nível)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Retorno diário (sempre disponível no dia seguinte)
        df['retorno_diario'] = df['Close'].pct_change()
        
        # Médias móveis - usando períodos mínimos adequados
        for janela in [5, 10, 20, 50]:
            df[f'MM_{janela}'] = df['Close'].rolling(window=janela, min_periods=min(janela, self.periodos_minimos)).mean()
        
        # Razões de médias móveis (indicam tendência)
        df['razao_MM_5_20'] = df['MM_5'] / df['MM_20']
        df['razao_MM_10_50'] = df['MM_10'] / df['MM_50']
        
        # Posição do preço em relação às médias móveis
        df['preco_vs_MM5'] = df['Close'] / df['MM_5'] - 1
        df['preco_vs_MM20'] = df['Close'] / df['MM_20'] - 1
        
        # Volatilidade histórica
        df['volatilidade_5'] = df['retorno_diario'].rolling(window=5, min_periods=5).std()
        df['volatilidade_10'] = df['retorno_diario'].rolling(window=10, min_periods=10).std()
        df['volatilidade_20'] = df['retorno_diario'].rolling(window=20, min_periods=20).std()
        
        # RSI (Índice de Força Relativa)
        delta = df['Close'].diff()
        ganho = delta.where(delta > 0, 0)
        perda = -delta.where(delta < 0, 0)
        
        ganho_medio = ganho.rolling(window=14, min_periods=14).mean()
        perda_media = perda.rolling(window=14, min_periods=14).mean()
        
        rs = ganho_medio / (perda_media + 1e-10)  # Evita divisão por zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Convergência e Divergência de Médias Móveis)
        exp12 = df['Close'].ewm(span=12, min_periods=12).mean()
        exp26 = df['Close'].ewm(span=26, min_periods=26).mean()
        df['MACD'] = exp12 - exp26
        df['sinal_MACD'] = df['MACD'].ewm(span=9, min_periods=9).mean()
        df['histograma_MACD'] = df['MACD'] - df['sinal_MACD']
        
        # Features de volume (se disponível)
        if 'Volume' in df.columns:
            df['volume_mm_10'] = df['Volume'].rolling(window=10, min_periods=10).mean()
            df['volume_mm_20'] = df['Volume'].rolling(window=20, min_periods=20).mean()
            df['razao_volume'] = df['Volume'] / df['volume_mm_20']
            df['tendencia_volume'] = df['volume_mm_10'] / df['volume_mm_20']
        
        # Lags de retornos (momentum histórico)
        for lag in [1, 2, 3, 5]:
            df[f'retorno_lag_{lag}'] = df['retorno_diario'].shift(lag)
        
        # Momentum de curto prazo
        df['momentum_3d'] = df['Close'] / df['Close'].shift(3) - 1
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        
        # Range normalizado (volatilidade intraday)
        df['range_normalizado'] = (df['High'] - df['Low']) / df['Close']
        
        return df
    
    def criar_targets(self, df):
        """
        Cria variáveis target APENAS após todas as features estarem prontas.
        
        Args:
            df (pandas.DataFrame): DataFrame com features já criadas
            
        Returns:
            pandas.DataFrame: DataFrame com targets e features avançadas adicionadas
        """
        df = df.copy()
        
        # Target para classificação: próximo retorno > 0
        df['target_classe'] = (df['retorno_diario'].shift(-1) > 0).astype(int)
        
        # Target para regressão: próximo preço
        df['target_preco'] = df['Close'].shift(-1)
        
        # Target para análise: próximo retorno
        df['target_retorno'] = df['retorno_diario'].shift(-1)
        
        # ==================== FEATURES AVANÇADAS PARA MELHOR TAXA DE ACERTO ====================
        # Suporte e Resistência dinâmicos
        df['nivel_suporte'] = df['Low'].rolling(window=20, min_periods=10).min()
        df['nivel_resistencia'] = df['High'].rolling(window=20, min_periods=10).max()
        df['distancia_suporte'] = (df['Close'] - df['nivel_suporte']) / (df['Close'] + 1e-10)
        df['distancia_resistencia'] = (df['nivel_resistencia'] - df['Close']) / (df['Close'] + 1e-10)
        
        # Momentum de múltiplas escalas OTIMIZADO
        for periodo in [3, 7, 14]:
            df[f'roc_{periodo}'] = ((df['Close'] / df['Close'].shift(periodo)) - 1) * 100
            df[f'roc_{periodo}_mm'] = df[f'roc_{periodo}'].rolling(window=3, min_periods=2).mean()
        
        # Volatilidade adaptativa MELHORADA
        df['regime_vol_curto'] = (df['volatilidade_5'] > df['volatilidade_5'].rolling(window=20, min_periods=10).median()).astype(int)
        df['breakout_vol'] = (df['volatilidade_5'] > df['volatilidade_5'].rolling(window=30, min_periods=15).quantile(0.75)).astype(int)
        
        # Sinais de reversão à média OTIMIZADOS
        df['sobrevendido'] = (df['RSI'] < 35).astype(int)  # Menos extremo
        df['sobrecomprado'] = (df['RSI'] > 65).astype(int)  # Menos extremo
        df['momentum_rsi'] = df['RSI'].diff(3)  # Momentum do RSI
        
        # Bandas de Bollinger OTIMIZADAS
        periodo_bb = 15  # Período menor para mais responsividade
        desvio_bb = 1.8    # Desvio menor para mais sinais
        df['bb_meio'] = df['Close'].rolling(window=periodo_bb, min_periods=periodo_bb//2).mean()
        bb_desvio_movel = df['Close'].rolling(window=periodo_bb, min_periods=periodo_bb//2).std()
        df['bb_superior'] = df['bb_meio'] + (bb_desvio_movel * desvio_bb)
        df['bb_inferior'] = df['bb_meio'] - (bb_desvio_movel * desvio_bb)
        df['posicao_bb'] = (df['Close'] - df['bb_inferior']) / (df['bb_superior'] - df['bb_inferior'] + 1e-10)
        df['largura_bb'] = (df['bb_superior'] - df['bb_inferior']) / df['bb_meio']
        
        # Features de FORÇA DA TENDÊNCIA
        df['forca_tendencia'] = abs(df['MM_5'] - df['MM_20']) / df['MM_20']
        df['momentum_preco'] = (df['Close'] / df['Close'].shift(5) - 1) * df['forca_tendencia']
        
        # Features de ANÁLISE DE VOLUME (se disponível)
        if 'Volume' in df.columns:
            df['tendencia_volume_preco'] = df['razao_volume'] * df['retorno_diario']
            df['breakout_volume'] = (df['Volume'] > df['Volume'].rolling(window=10, min_periods=5).quantile(0.8)).astype(int)
        
        return df

    def selecionar_melhores_features(self, X, y_classe, max_features=20):
        """
        Seleciona as melhores features usando múltiplos critérios estatísticos.
        
        Args:
            X (pandas.DataFrame): DataFrame com features candidatas
            y_classe (pandas.Series): Variável target para classificação
            max_features (int): Número máximo de features a selecionar
            
        Returns:
            list: Lista com nomes das features selecionadas
        """
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        # 1. Correlação com target
        correlacoes = abs(X.corrwith(y_classe)).sort_values(ascending=False)
        
        # 2. Informação Mútua
        scores_mi = mutual_info_classif(X, y_classe, random_state=42)
        features_mi = pd.Series(scores_mi, index=X.columns).sort_values(ascending=False)
        
        # 3. Importância do Random Forest
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
        rf_temp.fit(X, y_classe)
        importancia_rf = pd.Series(rf_temp.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # 4. Remove features altamente correlacionadas entre si
        matriz_corr = X.corr().abs()
        triangulo_superior = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
        features_alta_corr = [coluna for coluna in triangulo_superior.columns if any(triangulo_superior[coluna] > 0.8)]
        
        # 5. Combina rankings (média dos ranks)
        scores_features = pd.DataFrame({
            'correlacao': correlacoes.rank(ascending=False),
            'info_mutua': features_mi.rank(ascending=False),
            'importancia_rf': importancia_rf.rank(ascending=False)
        })
        
        scores_features['rank_medio'] = scores_features.mean(axis=1)
        scores_features = scores_features.sort_values('rank_medio')
        
        # 6. Seleciona top features, evitando alta correlação
        features_selecionadas = []
        for feature in scores_features.index:
            if feature not in features_alta_corr and len(features_selecionadas) < max_features:
                features_selecionadas.append(feature)
            elif len(features_selecionadas) < max_features:
                # Verifica correlação com features já selecionadas
                corr_maxima = 0
                if features_selecionadas:
                    corr_maxima = max([abs(X[feature].corr(X[feat_sel])) for feat_sel in features_selecionadas])
                
                if corr_maxima < 0.7:  # Limiar de correlação
                    features_selecionadas.append(feature)
        
        print(f"🎯 Seleção de Features: {len(X.columns)} → {len(features_selecionadas)} features")
        print(f"📊 Features selecionadas: {features_selecionadas}")
        
        return features_selecionadas

    def preparar_dados_treinamento(self, df):
        """
        Prepara dados para treinamento com seleção inteligente de features.
        
        Args:
            df (pandas.DataFrame): DataFrame com features e targets criados
            
        Returns:
            tuple: (X, y_classe, y_preco, y_retorno, features_selecionadas)
        """
        
        # Lista OTIMIZADA de features candidatas
        candidatas_features = [
            # Médias móveis e tendências
            'razao_MM_5_20', 'razao_MM_10_50', 'preco_vs_MM5', 'preco_vs_MM20',
            'forca_tendencia', 'momentum_preco',
            
            # Volatilidade
            'volatilidade_5', 'volatilidade_10', 'volatilidade_20',
            'regime_vol_curto', 'breakout_vol', 'range_normalizado',
            
            # Indicadores técnicos
            'RSI', 'momentum_rsi', 'sobrevendido', 'sobrecomprado',
            'MACD', 'sinal_MACD', 'histograma_MACD',
            
            # Bandas de Bollinger
            'posicao_bb', 'largura_bb',
            
            # Volume (se disponível)
            'razao_volume', 'tendencia_volume', 'tendencia_volume_preco', 'breakout_volume',
            
            # Momentum e lags
            'retorno_lag_1', 'retorno_lag_2', 'retorno_lag_3', 'retorno_lag_5',
            'momentum_3d', 'momentum_5d',
            'roc_3', 'roc_7', 'roc_14', 'roc_3_mm', 'roc_7_mm',
            
            # Suporte/Resistência
            'distancia_suporte', 'distancia_resistencia'
        ]
        
        # Filtra apenas features que existem no DataFrame
        features_disponiveis = [f for f in candidatas_features if f in df.columns]
        
        if len(features_disponiveis) < 10:
            print(f"⚠️  Apenas {len(features_disponiveis)} features disponíveis de {len(candidatas_features)} candidatas")
            # Adiciona features básicas se necessário
            features_basicas = [
                'razao_MM_5_20', 'razao_MM_10_50', 'preco_vs_MM5', 'preco_vs_MM20',
                'volatilidade_5', 'volatilidade_10', 'volatilidade_20',
                'RSI', 'MACD', 'sinal_MACD', 'histograma_MACD',
                'razao_volume', 'tendencia_volume',
                'retorno_lag_1', 'retorno_lag_2', 'retorno_lag_3', 'retorno_lag_5',
                'momentum_3d', 'momentum_5d', 'range_normalizado'
            ]
            for feature in features_basicas:
                if feature in df.columns and feature not in features_disponiveis:
                    features_disponiveis.append(feature)
        
        candidatas_features = features_disponiveis
        candidatas_disponiveis = [f for f in candidatas_features if f in df.columns]
        
        # Remove linhas com NaN
        colunas_obrigatorias = candidatas_disponiveis + ['target_classe', 'target_preco', 'target_retorno']
        df_limpo = df.dropna(subset=colunas_obrigatorias)
        
        amostras_minimas = 100 if 'target_classe' in df_limpo.columns and len(df_limpo) > 100 else 30
        if len(df_limpo) < amostras_minimas:
            raise ValueError(f"Dados insuficientes após limpeza: {len(df_limpo)} registros")
        
        # Separa features candidatas e targets
        X_candidatas = df_limpo[candidatas_disponiveis].copy()
        y_classe = df_limpo['target_classe'].copy()
        y_preco = df_limpo['target_preco'].copy()
        y_retorno = df_limpo['target_retorno'].copy()
        
        # Tratamento de valores infinitos e NaN
        X_candidatas = X_candidatas.replace([np.inf, -np.inf], np.nan)
        X_candidatas = X_candidatas.fillna(X_candidatas.median())
        
        # Seleção inteligente de features
        features_selecionadas = self.selecionar_melhores_features(X_candidatas, y_classe, max_features=10)
        
        # Dataset final com features selecionadas
        X = X_candidatas[features_selecionadas].copy()
        
        # Validação final
        mascara = ~(X.isnull().any(axis=1) | y_classe.isnull() | y_preco.isnull() | y_retorno.isnull())
        
        X = X[mascara]
        y_classe = y_classe[mascara]
        y_preco = y_preco[mascara]
        y_retorno = y_retorno[mascara]
        
        # Normalização robusta (remove outliers)
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            X[col] = X[col].clip(limite_inferior, limite_superior)
        
        print(f"✅ Dados preparados: {len(X)} amostras, {len(features_selecionadas)} features")
        print(f"📊 Período: {X.index.min().strftime('%Y-%m-%d')} a {X.index.max().strftime('%Y-%m-%d')}")
        print(f"📈 Features selecionadas: {features_selecionadas}")
        
        return X, y_classe, y_preco, y_retorno, features_selecionadas
