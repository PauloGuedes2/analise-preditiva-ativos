import atexit
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from joblib import load
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.config.params import Params
from src.data.data_loader import DataLoader
from src.data.data_updater import data_updater
from src.models.feature_engineer import FeatureEngineer
from src.backtesting.risk_analyzer import RiskAnalyzer

# Configuração da página do Streamlit
st.set_page_config(layout="wide", page_title="Análise Preditiva de Ativos", page_icon="📈")


class DashboardTrading:
    """Dashboard Streamlit para Análise Preditiva de Ativos"""

    def __init__(self):
        """Inicializa o dashboard, o serviço de atualização de dados e a barra lateral."""
        self.modelo_carregado = None
        self.ticker_selecionado = None
        data_updater.iniciar_atualizacao_automatica(tickers=Params.TICKERS)
        self._inicializar_sidebar()

    def _inicializar_sidebar(self):
        """Configura e renderiza a barra lateral com as opções de seleção."""
        with st.sidebar:
            st.markdown("## 📈 Análise Preditiva")
            st.markdown("---")
            # Procura por arquivos de modelo no diretório configurado
            modelos_disponiveis = sorted(
                [f.replace('modelo_', '').replace('.joblib', '') for f in os.listdir(Params.PATH_MODELOS) if
                 f.endswith('.joblib')])
            if not modelos_disponiveis:
                st.warning("Nenhum modelo treinado foi encontrado.")
                st.stop() # Encerra a execução se não houver modelos

            self.ticker_selecionado = st.selectbox("Selecione o Ativo:", modelos_disponiveis,
                                                   help="Escolha um dos modelos previamente treinados para análise.")
            st.markdown("---")
            st.markdown("##### Escolha o tipo de análise:")
            self.analisar_btn = st.button("🔍 Análise Interativa (Dashboard)", use_container_width=True)
            self.relatorio_btn = st.button("📋 Gerar Relatório Completo", type="primary", use_container_width=True)
            st.markdown("---")

            with st.expander("Manutenção e Ajuda"):
                if st.button("🔄 Forçar Reset dos Dados", use_container_width=True,
                             help="Apaga o banco de dados local para forçar o download de dados novos na próxima análise."):
                    self._forcar_download_dados()

    @st.cache_resource(ttl=3600, show_spinner="Carregando modelo...")
    def _carregar_modelo(_self, ticker: str) -> Optional[Any]:
        """
        Carrega o modelo treinado do disco, com cache para performance.

        Args:
            ticker (str): O ticker do modelo a ser carregado.

        Returns:
            Optional[Any]: O objeto do modelo carregado ou None se não encontrado.
        """
        caminho = os.path.join(Params.PATH_MODELOS, f"modelo_{ticker}.joblib")
        if os.path.exists(caminho):
            try:
                return load(caminho)
            except Exception as e:
                st.error(f"Erro ao carregar o modelo '{ticker}': {e}")
                return None
        return None

    def executar(self):
        """Função principal que orquestra a lógica e renderização do dashboard."""
        if self.analisar_btn or self.relatorio_btn:
            self.modelo_carregado = self._carregar_modelo(self.ticker_selecionado)
            if self.modelo_carregado is None:
                st.error(f"O modelo para {self.ticker_selecionado} não foi encontrado.")
                return
            with st.spinner("Processando dados e gerando análise..."):
                df_ticker, df_ibov, X_full, y_full, precos_full, previsao = self._processar_dados_e_previsao()

            # Decide qual tipo de visualização renderizar
            if self.relatorio_btn:
                self._render_relatorio_completo(previsao, X_full, y_full, precos_full, df_ibov, df_ticker)
            else:
                self._render_analise_em_abas(previsao, X_full, y_full, precos_full, df_ibov, df_ticker)
        else:
            self._render_tela_boas_vindas()

    def _processar_dados_e_previsao(self):
        """Baixa, processa os dados e gera a previsão usando o modelo carregado."""
        loader = DataLoader()
        feature_engineer = FeatureEngineer()
        try:
            df_ticker, df_ibov = loader.baixar_dados_yf(self.ticker_selecionado)
        except Exception as e:
            st.warning(
                f"**Aviso:** Falha ao baixar dados ({e}). Usando a última versão salva no banco de dados local.")
            df_ticker = loader.carregar_do_bd(self.ticker_selecionado)
            df_ibov = loader.carregar_do_bd('^BVSP')
        if df_ticker.empty:
            st.error(f"Não foi possível carregar dados para {self.ticker_selecionado}.")
            st.stop()
        X_full, y_full, precos_full, _, X_untruncated = feature_engineer.preparar_dataset(df_ticker, df_ibov, self.ticker_selecionado)
        previsao = self.modelo_carregado.prever_direcao(X_untruncated.tail(1), self.ticker_selecionado)
        return df_ticker, df_ibov, X_full, y_full, precos_full, previsao

    @staticmethod
    def _render_tela_boas_vindas():
        """Renderiza a tela inicial de boas-vindas com informações sobre o projeto."""
        st.title("Bem-vindo ao Sistema de Análise Preditiva")
        st.markdown("---")
        st.header("Entendendo o Modelo: Um Guia Rápido")
        st.info("**Selecione um ativo na barra lateral e escolha um tipo de análise para começar.**", icon="👈")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 O Que é Este Projeto?")
            st.write(
                "Este é um sistema de **apoio à decisão** baseado em Machine Learning. Seu objetivo **não é** dar recomendações de compra ou venda, mas sim **identificar oportunidades potenciais** de alta para um ativo, com base em padrões históricos.")
            st.subheader("🧠 Como o Modelo 'Pensa'?")
            st.write(
                "Utilizamos um modelo de **Classificação** que prevê a **direção** do movimento. O alvo é definido pela **Metodologia da Tripla Barreira**, que considera uma janela de tempo futura para determinar se uma operação teria sido um sucesso (lucro), um fracasso (stop) ou neutra.")
        with col2:
            st.subheader("🔬 Como a Confiança é Medida?")
            st.write(
                "A performance do modelo é avaliada pelo método de **Validação Walk-Forward (WFV)**. Este processo simula a operação em tempo real, representando uma estimativa muito mais honesta de seu desempenho.")
            st.subheader("⚠️ Limitações e Boas Práticas")
            st.write(
                "- **Performance Passada Não Garante Futuro:** O mercado é dinâmico e padrões podem não se repetir.\n"
                "- **Não é uma Bola de Cristal:** Fatores macroeconômicos e notícias não estão no escopo do modelo.\n"
                "- **Use como Ferramenta:** Esta análise deve ser usada como mais uma camada de informação em seu processo de decisão.")

    def _render_analise_em_abas(self, previsao, X_full, y_full, precos_full, df_ibov, df_ticker):
        """Renderiza a análise preditiva em um layout de abas."""
        st.header(f"Análise Preditiva para {self.ticker_selecionado}")
        tabs = st.tabs(["🎯 **Resumo Executivo**", "🔍 **Análise da Previsão**", "🩺 **Saúde do Modelo**",
                        "📈 **Análise de Mercado**", "🧬 **DNA do Modelo**", "📊 **Simulação de Performance**"])

        with tabs[0]: self._render_tab_resumo(previsao, precos_full, df_ticker)
        with tabs[1]: self._render_tab_previsao_shap(X_full)
        with tabs[2]: self._render_tab_saude_modelo(X_full)
        with tabs[3]: self._render_tab_mercado(precos_full, df_ibov)
        with tabs[4]: self._render_tab_dna(y_full)
        with tabs[5]: self._render_tab_simulacao(X_full, precos_full)

    def _render_relatorio_completo(self, previsao, X_full, y_full, precos_full, df_ibov, df_ticker):
        """Renderiza um relatório completo de análise preditiva."""
        st.title(f"📋 Relatório de Análise Preditiva: {self.ticker_selecionado}")

        last_date = df_ticker.index[-1].strftime('%d/%m/%Y')
        next_date = (df_ticker.index[-1] + pd.tseries.offsets.BDay(1)).strftime('%d/%m/%Y')

        st.caption(
            f"Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} | Previsão para o pregão de {next_date}")
        st.header("1. Conclusão Executiva: Qual é o Veredito?")
        recomendacao = "🟢 **OPORTUNIDADE**" if previsao['should_operate'] else "🟡 **OBSERVAR**"
        probabilidade = previsao['probabilidade']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"Para o próximo pregão, o modelo sinaliza **{recomendacao}** para o ativo **{self.ticker_selecionado}**.")

            st.metric("Data da Previsão", next_date)

            st.metric("Confiança do Modelo na Alta", f"{probabilidade:.1%}",
                      help="Probabilidade estimada pelo modelo para um movimento de alta, conforme definido pela estratégia da Tripla Barreira.")
            st.metric("Limiar Mínimo para Operar", f"{self.modelo_carregado.threshold_operacional:.1%}",
                      help="O modelo só recomenda 'Oportunidade' se a confiança superar este valor, que foi calibrado para otimizar a performance.")
        with col2:
            self._plot_gauge_confianca(probabilidade)
        st.header("2. Diagnóstico de Confiança: Por que Confiar Nesta Previsão?")
        st.info(
            "A confiança na previsão não é arbitrária. Ela se baseia no desempenho histórico robusto do modelo, validado através do método **Walk-Forward**, que simula como o modelo teria performado em condições reais no passado.")
        self._render_diagnostico_confianca()
        st.header("3. Contexto de Mercado: Ativo vs. IBOVESPA")
        st.write(
            "A performance do ativo é comparada com o índice Bovespa para avaliar seu desempenho relativo ao mercado como um todo.")
        self._plot_performance_vs_ibov(precos_full, df_ibov)
        st.header("4. O 'Cérebro' do Modelo: Como a Decisão foi Tomada?")
        st.write(
            "O modelo não é uma 'caixa-preta'. Abaixo, vemos os fatores que ele mais considerou para a sua decisão e a prova de sua capacidade de classificação.")
        col_dna1, col_dna2 = st.columns(2)
        with col_dna1:
            st.markdown("**Variáveis Mais Influentes**")
            self._plot_importancia_features()
        with col_dna2:
            st.markdown("**Prova de Performance (Classificação)**")
            self._plot_matriz_confusao(y_full)
        self._render_traducao_features()
        st.header("5. Metodologia e Glossário")
        self._render_glossario_metodologia()

    def _render_tab_resumo(self, previsao: Dict[str, Any], precos_full: pd.Series,
                           df_ticker: pd.DataFrame):
        """Renderiza a aba de resumo executivo com a previsão e contexto recente do ativo."""
        st.subheader("Diagnóstico e Previsão")
        col1, col2 = st.columns([2, 1])
        with col1:
            self._render_diagnostico_confianca()
        with col2:
            recomendacao = "🟢 **OPORTUNIDADE**" if previsao['should_operate'] else "🟡 **OBSERVAR**"
            st.markdown("##### Sinal para o Próximo Pregão")
            st.markdown(f"<h1>{recomendacao}</h1>", unsafe_allow_html=True)

            proximo_dia_util = (df_ticker.index[-1] + pd.tseries.offsets.BDay(1)).strftime('%d/%m/%Y')

            st.metric("Data da Previsão", proximo_dia_util)

            probabilidade = previsao['probabilidade']
            st.progress(probabilidade, text=f"{probabilidade:.1%} de Confiança na Alta")
        st.divider()
        st.subheader("Previsão no Contexto do Preço Recente")
        self._plot_previsao_recente(precos_full, previsao['should_operate'])

    def _render_tab_mercado(self, precos_full: pd.Series, df_ibov: pd.DataFrame):
        """Renderiza a aba de análise de mercado com a performance do ativo versus o IBOVESPA."""
        st.subheader("Performance Relativa: Ativo vs. IBOVESPA (Último Ano)")
        st.info(
            "Este gráfico compara o crescimento de R$100 investidos no ativo selecionado versus no índice IBOVESPA. Ele ajuda a responder: a ação está com desempenho melhor ou pior que a média do mercado?")
        self._plot_performance_vs_ibov(precos_full, df_ibov)

    def _render_tab_dna(self, y_full: pd.Series):
        """Renderiza a aba de DNA do modelo com importância das features e matriz de confusão."""
        st.subheader("O 'Cérebro' do Modelo: Fatores de Decisão")
        st.info("Aqui exploramos o que o modelo considera importante e como sua performance foi validada.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Variáveis Mais Influentes**")
            st.caption("O que o modelo 'olha' para tomar a decisão.")
            self._plot_importancia_features()
        with col2:
            st.markdown("**Performance de Classificação (Última Validação)**")
            st.caption("Como o modelo se saiu ao classificar os cenários.")
            self._plot_matriz_confusao(y_full)
        self._render_traducao_features()

    def _render_tab_simulacao(self, X_full: pd.DataFrame, precos_full: pd.Series):
        """Renderiza a aba de simulação de performance com backtest e análise de risco."""
        st.warning(
            "Esta simulação é **'in-sample'** e tende a ser otimista. Serve principalmente para **ilustrar o comportamento visual** da estratégia ao longo do tempo.")
        risk_analyzer = RiskAnalyzer()
        df_sinais = self.modelo_carregado.prever_e_gerar_sinais(X_full, precos_full, self.ticker_selecionado)
        backtest_info = risk_analyzer.backtest_sinais(df_sinais)

        st.plotly_chart(self._plot_precos_sinais(df_sinais, precos_full), use_container_width=True)

        if backtest_info.get('trades', 0) > 0:
            self._exibir_metricas_backtest(backtest_info)
            st.divider()
            st.subheader("Análise de Risco e Capital")
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(self._plot_equidade(backtest_info), use_container_width=True)
            with col2: self._plot_drawdown_curve(backtest_info)

            with st.expander("Análise de Sensibilidade do Threshold de Operação"):
                st.info(
                    "Este gráfico mostra como a performance (Sharpe Ratio) da estratégia mudaria com diferentes limiares de confiança. Um pico largo em torno do threshold escolhido (linha vermelha) indica uma estratégia robusta.")
                with st.spinner("Analisando sensibilidade do threshold..."):
                    df_sensibilidade = self._analisar_sensibilidade_threshold(X_full, precos_full)
                    self._plot_sensibilidade_threshold(df_sensibilidade)

    def _render_tab_previsao_shap(self, X_full: pd.DataFrame):
        """Renderiza a aba de explicabilidade da previsão atual usando SHAP."""
        st.subheader("Explicabilidade da Previsão Atual (SHAP)")
        st.info(
            "Este gráfico de cascata (waterfall) mostra como cada variável (feature) contribuiu para a previsão final. Features em vermelho empurram a previsão para cima (mais chance de 'Oportunidade'), enquanto as azuis empurraram para baixo.")

        if not hasattr(self.modelo_carregado, 'shap_explainer') or self.modelo_carregado.shap_explainer is None:
            st.warning(
                "O explainer SHAP não foi encontrado neste modelo. É necessário retreinar o modelo para gerar esta análise.")
            return

        with st.spinner("Calculando valores SHAP..."):
            X_last = X_full[self.modelo_carregado.features_selecionadas].tail(1)
            X_last_scaled = self.modelo_carregado.scaler.transform(X_last)

            # Obter os valores SHAP para a última amostra.
            shap_explanation_raw = self.modelo_carregado.shap_explainer(X_last_scaled)

            # Identificar o índice da classe 'Alta' (1) no array de classes do label encoder.
            idx_classe_1 = np.where(self.modelo_carregado.label_encoder.classes_ == 1)[0][0]

            # Fatiar o objeto SHAP para obter apenas a explicação da classe 'Alta'.
            shap_explanation_for_plot = shap_explanation_raw[0, :, idx_classe_1]

            # Plotar o gráfico de cascata (waterfall) usando a explicação fatiada.
            shap.initjs()
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_explanation_for_plot, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    def _render_tab_saude_modelo(self, X_full: pd.DataFrame):
        """Renderiza a aba de monitoramento de saúde do modelo, focando em data drift."""
        st.subheader("Monitoramento de Saúde do Modelo (Data Drift)")
        st.info(
            "Aqui comparamos as características dos dados mais recentes com os dados usados para treinar o modelo. Diferenças significativas (drift) podem indicar que o modelo precisa ser retreinado.")

        if not hasattr(self.modelo_carregado,
                       'training_data_profile') or self.modelo_carregado.training_data_profile is None:
            st.warning("O perfil dos dados de treino não foi encontrado. Retreine o modelo para gerar esta análise.")
            return

        baseline_profile = pd.DataFrame(self.modelo_carregado.training_data_profile)
        X_recent = X_full[self.modelo_carregado.features_selecionadas].tail(60)
        X_recent_scaled = pd.DataFrame(self.modelo_carregado.scaler.transform(X_recent), columns=X_recent.columns)
        current_profile = X_recent_scaled.describe()

        st.markdown("**Comparativo Estatístico (Dados de Treino vs. Dados Recentes)**")
        st.dataframe(pd.concat([baseline_profile, current_profile], axis=1,
                               keys=['Treino (Baseline)', 'Recente (Últimos 60 dias)']))

        st.markdown("**Análise Visual de Drift das Features Principais**")
        key_features = ['rsi_14', 'vol_21d', 'sma_ratio_10_50', 'momentum_5d']
        cols = st.columns(len(key_features))

        for i, feature in enumerate(key_features):
            if feature in baseline_profile.columns and feature in X_recent_scaled.columns:
                with cols[i]:
                    self._plot_drift_distribution(baseline_profile[feature], X_recent_scaled[feature], feature)

    @st.cache_data
    def _analisar_sensibilidade_threshold(_self, X: pd.DataFrame, precos: pd.Series) -> pd.DataFrame:
        """Analisa a sensibilidade do threshold de operação variando-o e calculando o Sharpe Ratio."""
        risk_analyzer = RiskAnalyzer()
        resultados = []
        for thr in np.arange(0.40, 0.75, 0.025):
            df_sinais = _self.modelo_carregado.prever_e_gerar_sinais(X, precos, _self.ticker_selecionado,
                                                                     threshold_override=thr)
            backtest = risk_analyzer.backtest_sinais(df_sinais, verbose=False)
            resultados.append({'threshold': thr, 'sharpe': backtest.get('sharpe', 0)})
        return pd.DataFrame(resultados)

    def _plot_sensibilidade_threshold(self, df_sensibilidade: pd.DataFrame):
        """Plota a análise de sensibilidade do threshold de operação versus o Sharpe Ratio."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sensibilidade['threshold'], y=df_sensibilidade['sharpe'], mode='lines+markers',
                                 name='Sharpe Ratio'))
        fig.add_vline(x=self.modelo_carregado.threshold_operacional, line_width=2, line_dash="dash", line_color="red",
                      annotation_text="Threshold Escolhido", annotation_position="top left")
        fig.update_layout(title="Performance (Sharpe) vs. Threshold de Confiança",
                          xaxis_title="Threshold de Confiança", yaxis_title="Sharpe Ratio Anualizado")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _plot_drift_distribution(baseline_series: pd.Series, current_series: pd.Series, feature_name: str):
        """"Plota a distribuição das features para detectar data drift usando o teste KS."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=baseline_series, name='Treino', opacity=0.75, histnorm='probability density'))
        fig.add_trace(go.Histogram(x=current_series, name='Recente', opacity=0.75, histnorm='probability density'))
        fig.update_layout(barmode='overlay', title_text=f'Drift de {feature_name}', height=300,
                          margin=dict(t=30, b=10, l=10, r=10))

        ks_stat, p_value = ks_2samp(baseline_series.dropna(), current_series.dropna())
        st.plotly_chart(fig, use_container_width=True)

        if p_value < 0.05:
            st.error(f"P-valor (KS): {p_value:.3f}. Drift significativo detectado!")
        else:
            st.success(f"P-valor (KS): {p_value:.3f}. Sem drift significativo.")

    def _render_diagnostico_confianca(self):
        """Renderiza o diagnóstico de confiança do modelo baseado nas métricas históricas."""
        st.markdown("##### Diagnóstico de Confiança do Modelo (Histórico WFV)")
        metricas = self.modelo_carregado.wfv_metrics
        score, max_score, confianca_txt, cor = self._calcular_indice_confiabilidade(metricas)
        percentual = (score / max_score) * 100
        fig = go.Figure(go.Indicator(mode="gauge+number", value=percentual,
                                     title={
                                         'text': f"<span style='font-size:1.5em;color:{cor}'>{confianca_txt}</span>"},
                                     gauge={'axis': {'range': [None, 100]}, 'bar': {'color': cor}}))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
        col1, col2 = st.columns([1, 2])
        with col1: st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Score de Confiança", f"{score} / {max_score}",
                      help="Pontuação baseada na performance histórica. Quanto maior, mais robusto o modelo se provou em testes.")
            st.metric("Sharpe Médio", f"{metricas.get('sharpe_medio', 0):.2f}",
                      help="Mede o retorno ajustado ao risco. Acima de 0.5 é bom, acima de 1.0 é ótimo.")
            st.metric("F1-Score Preditivo", f"{metricas.get('f1_macro_medio', 0):.2%}",
                      help="Mede a qualidade das previsões. Acima de 60% indica boa capacidade preditiva.")
        with st.expander("Como este Score é Calculado?"):
            st.markdown("""
            O score é a soma de pontos baseados em 3 pilares da performance histórica do modelo (validação Walk-Forward):
            - **Risco-Retorno (Sharpe Ratio):** `> 1.0`: **+3 pontos** (Excelente) | `> 0.3`: **+2 pontos** (Bom) | `> -0.1`: **+1 ponto** (Aceitável)
            - **Qualidade Preditiva (F1-Score):** `> 65%`: **+3 pontos** (Ótima) | `> 55%`: **+2 pontos** (Boa) | `> 50%`: **+1 ponto** (Razoável)
            - **Frequência de Trades:** `> 8 trades`: **+3 pontos** (Ativo) | `> 4 trades`: **+2 pontos** (Moderado) | `> 2.5 trades`: **+1 ponto** (Seletivo)
            **Score Final:** `7-9`: **Alta Confiança** | `4-6`: **Média Confiança** | `0-3`: **Baixa Confiança**
            """)

    @staticmethod
    def _render_traducao_features():
        """Renderiza a tradução das features técnicas para termos simples."""
        with st.expander("O que esses fatores significam em termos simples? 🤔"):
            st.markdown("""
            - **`rsi_14` (Índice de Força Relativa):** Mede se o ativo está "caro" (sobrecomprado) ou "barato" (sobrevendido) recentemente. Ajuda a identificar possíveis pontos de virada.
            - **`sma_ratio` (Razão de Médias Móveis):** Compara uma tendência de curto prazo com uma de longo prazo. Se a curta está acima da longa, indica uma tendência de alta.
            - **`vol_21d` (Volatilidade):** Mede o "grau de agitação" do preço. Alta volatilidade pode significar tanto risco quanto oportunidade.
            - **`momentum`:** Mede a velocidade e a força do movimento dos preços em um período.
            - **`correlacao_ibov`:** Indica se o ativo tende a se mover junto com o índice Bovespa ou na direção contrária.
            """)

    @staticmethod
    def _render_glossario_metodologia():
        """Renderiza o glossário e explicações sobre a metodologia usada no modelo."""
        with st.expander("Glossário: Entendendo os Termos Técnicos 📖"):
            st.markdown("""
            - **Walk-Forward Validation (WFV):** A espinha dorsal da confiança neste modelo. Em vez de testar o modelo em dados que ele já 'espiou' durante o treino, o WFV simula a passagem do tempo: o modelo treina com dados do passado (ex: 2022) e é testado em dados do 'futuro' que ele nunca viu (ex: 2023). Isso resulta em uma estimativa de performance muito mais realista e confiável.
            - **Sharpe Ratio:** A métrica mais importante para avaliar uma estratégia de investimento. Ela não mede apenas o retorno, mas o **retorno ajustado ao risco**. Um Sharpe Ratio alto (acima de 1.0 é excelente) significa que a estratégia gera bons retornos sem muita 'montanha-russa' no capital.
            - **F1-Score:** Uma métrica de Machine Learning que mede o equilíbrio entre 'acertar as oportunidades' (precisão) e 'não deixar oportunidades passarem' (recall). É mais robusta que a simples acurácia em mercados financeiros, onde os eventos de alta podem ser mais raros.
            - **Tripla Barreira:** O método usado para definir o que é um 'sucesso' ou 'fracasso'. Para cada dia, criamos três 'barreiras' no futuro (ex: 5 dias): uma de lucro (take profit), uma de perda (stop loss) e uma de tempo. O resultado da operação (alta, baixa ou neutro) é definido por qual barreira é tocada primeiro. Isso cria um alvo de previsão muito mais realista do que simplesmente 'o preço vai subir ou cair amanhã?'.
            """)
        st.warning(
            "⚠️ **Aviso Legal:** Esta é uma ferramenta de estudo e análise baseada em modelos estatísticos. A performance passada não é garantia de resultados futuros. Isto **não** constitui uma recomendação de investimento.")

    @staticmethod
    def _calcular_indice_confiabilidade(metricas: Dict[str, Any]) -> tuple[int, int, str, str]:
        """Calcula o índice de confiabilidade do modelo baseado em métricas históricas."""
        score, max_score = 0, 9
        sharpe = metricas.get('sharpe_medio', 0)
        f1 = metricas.get('f1_macro_medio', 0)
        trades = metricas.get('trades_medio', 0)
        if sharpe > 1.0:
            score += 3
        elif sharpe > 0.3:
            score += 2
        elif sharpe > -0.1:
            score += 1
        if f1 > 0.65:
            score += 3
        elif f1 > 0.55:
            score += 2
        elif f1 > 0.50:
            score += 1
        if trades > 8:
            score += 3
        elif trades > 4:
            score += 2
        elif trades >= 2.5:
            score += 1
        if score >= 7: return score, max_score, "Alta", "green"
        if score >= 4: return score, max_score, "Média", "orange"
        return score, max_score, "Baixa", "red"

    def _plot_gauge_confianca(self, probabilidade):
        """Plota um gauge mostrando a confiança do modelo na previsão atual."""
        threshold = self.modelo_carregado.threshold_operacional
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probabilidade * 100,
            title={'text': "Confiança na Alta (%)"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#007bff"},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75,
                                 'value': threshold * 100}}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A linha vermelha indica o limiar de confiança mínimo para operar.")

    @staticmethod
    def _plot_previsao_recente(precos: pd.Series, sinal_positivo: bool):
        """Plota o preço recente do ativo com a tendência e o sinal de previsão."""
        df_recente = precos.tail(90).copy().to_frame(name='Preço')
        df_recente['Tendência (20d)'] = df_recente['Preço'].rolling(20).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recente.index, y=df_recente['Preço'], mode='lines', name='Preço de Fechamento'))
        fig.add_trace(go.Scatter(x=df_recente.index, y=df_recente['Tendência (20d)'], mode='lines',
                                 name='Tendência (20 dias)',
                                 line={'dash': 'dot', 'color': 'gray'}))
        if sinal_positivo:
            ultimo_preco = df_recente['Preço'].iloc[-1]
            proximo_dia = df_recente.index[-1] + pd.tseries.offsets.BDay(1)
            fig.add_trace(go.Scatter(x=[proximo_dia], y=[ultimo_preco], mode='markers', name='Sinal de Oportunidade',
                                     marker=dict(color='green', size=15, symbol='circle',
                                                 line={'width': 2, 'color': 'darkgreen'})))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    def _plot_performance_vs_ibov(self, precos_ativo: pd.Series, df_ibov: pd.DataFrame):
        """Plota a performance do ativo versus o IBOVESPA, normalizando ambos para 100 no início do período."""
        if df_ibov is None or df_ibov.empty:
            st.warning("Não foi possível carregar os dados do IBOVESPA para comparação.")
            return
        if 'Close_IBOV' in df_ibov.columns:
            ibov_close = df_ibov['Close_IBOV']
        elif 'Close' in df_ibov.columns:
            ibov_close = df_ibov['Close']
        else:
            st.error("Coluna de fechamento do IBOVESPA não encontrada.")
            return
        df_comp = pd.DataFrame(precos_ativo).rename(columns={'Close': 'Ativo'})
        df_comp['IBOV'] = ibov_close
        df_comp = df_comp.dropna().tail(252)
        if df_comp.empty or len(df_comp) < 2:
            st.warning("Não há dados suficientes para a comparação entre o ativo e o IBOVESPA.")
            return
        df_normalizado = (df_comp / df_comp.iloc[0]) * 100
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df_normalizado.index, y=df_normalizado['Ativo'], mode='lines', name=self.ticker_selecionado))
        fig.add_trace(go.Scatter(x=df_normalizado.index, y=df_normalizado['IBOV'], mode='lines', name='IBOVESPA',
                                 line={'dash': 'dot', 'color': 'gray'}))
        fig.update_layout(title_text='Performance Normalizada (Base 100)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _plot_drawdown_curve(backtest_info: Dict[str, Any]):
        """Plota a curva de drawdown do backtest."""
        drawdown_series = backtest_info.get('drawdown_series', [])
        if not drawdown_series: return
        df_dd = pd.DataFrame(drawdown_series, columns=['Drawdown'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_dd.index, y=df_dd['Drawdown'], fill='tozeroy', name='Drawdown', line_color='red'))
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title_text='Curva de Drawdown da Estratégia', xaxis_title='Nº da Operação',
                          yaxis_title='Queda do Pico', height=350)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_matriz_confusao(self, y_full: pd.Series):
        """Plota a matriz de confusão do último fold da validação walk-forward."""
        try:
            X_full_scaled = self.modelo_carregado.X_scaled
            y_full_encoded = self.modelo_carregado.label_encoder.transform(y_full)
            _, test_idx = list(self.modelo_carregado.cv_gen.split(X_full_scaled))[-1]
            X_test, y_test_encoded = X_full_scaled.iloc[test_idx], y_full_encoded[test_idx]
            y_test_labels = self.modelo_carregado.label_encoder.inverse_transform(y_test_encoded)
            preds_labels = self.modelo_carregado.label_encoder.inverse_transform(
                self.modelo_carregado.modelo_final.predict(X_test))
            cm = confusion_matrix(y_test_labels, preds_labels, labels=[-1, 0, 1])
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BAIXA', 'NEUTRO', 'ALTA']).plot(ax=ax,
                                                                                                         cmap='Blues',
                                                                                                         values_format='d')
            ax.set_title("Previsões vs. Realidade")
            st.pyplot(fig)
            st.caption("Matriz de confusão do último 'fold' da validação.")
        except Exception as e:
            st.warning(f"Não foi possível gerar a Matriz de Confusão: {e}")

    @staticmethod
    def _forcar_download_dados():
        """Força o download dos dados novamente, apagando o banco de dados local."""
        st.info("Parando serviço de atualização para liberar o banco de dados...", icon="⏳")
        data_updater.parar_atualizacao()
        time.sleep(1)
        db_path = Params.PATH_DB_MERCADO
        try:
            if os.path.exists(db_path): os.remove(db_path)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Dados resetados com sucesso! A aplicação será recarregada.")
            time.sleep(2)
            st.rerun()
        except PermissionError:
            st.error("Não foi possível acessar o arquivo do banco de dados. Tente novamente em alguns segundos.")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")

    def _plot_importancia_features(self):
        """Plota a importância das features do modelo."""
        try:
            importances = self.modelo_carregado.modelo_final.feature_importances_
            features = self.modelo_carregado.features_selecionadas
            df_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance',
                                                                                                ascending=False).head(
                10)
            df_imp = df_imp.sort_values('importance', ascending=True)
            fig = go.Figure(go.Bar(x=df_imp['importance'], y=df_imp['feature'], orientation='h'))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico de importância: {e}")

    @staticmethod
    def _plot_precos_sinais(df_sinais, precos):
        """Plota os preços históricos com os sinais de operação gerados na simulação."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precos.index, y=precos, mode='lines', name='Preço'))
        sinais_operar = df_sinais[df_sinais['sinal'] == 1]
        if not sinais_operar.empty:
            fig.add_trace(
                go.Scatter(x=sinais_operar.index, y=sinais_operar['preco'], mode='markers', name='Oportunidade',
                           marker=dict(color='limegreen', size=6, symbol='triangle-up')))
        fig.update_layout(title_text='Preços Históricos e Sinais Gerados na Simulação',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig

    @staticmethod
    def _exibir_metricas_backtest(metricas: Dict[str, Any]):
        """Exibe as principais métricas do backtest da simulação."""
        st.subheader("Métricas de Performance da Simulação")
        cols = st.columns(4)
        cols[0].metric("Retorno Total", f"{metricas.get('retorno_total', 0):.2%}")
        cols[1].metric("Sharpe Ratio", f"{metricas.get('sharpe', 0):.2f}",
                       help="Mede o retorno ajustado por toda a volatilidade (risco total).")
        cols[2].metric("Sortino Ratio", f"{metricas.get('sortino', 0):.2f}",
                       help="Mede o retorno ajustado pela volatilidade negativa (risco de perdas).")
        cols[3].metric("Nº de Trades", f"{metricas.get('trades', 0)}")
        st.subheader("Métricas de Qualidade dos Trades")
        col_q1, col_q2, col_q3 = st.columns(3)
        col_q1.metric("Taxa de Acerto", f"{metricas.get('win_rate', 0):.2%}")
        col_q2.metric("Profit Factor", f"{metricas.get('profit_factor', 0):.2f}",
                      help="Soma dos lucros / Soma das perdas. > 1.5 é bom, > 2.0 é ótimo.")
        col_q3.metric("Payoff Ratio", f"{metricas.get('payoff_ratio', 0):.2f}",
                      help="Ganho médio / Perda média. > 1.5 indica que os ganhos compensam as perdas.")
        st.metric("Max Drawdown", f"{metricas.get('max_drawdown', 0):.2%}",
                  help="A maior queda percentual do capital a partir de um pico.")

    @staticmethod
    def _plot_equidade(backtest_info: Dict[str, Any]) -> go.Figure:
        """Plota a curva de equidade do backtest da simulação."""
        curva_equidade = backtest_info.get('equity_curve', [])
        fig = go.Figure()
        if len(curva_equidade) > 1:
            fig.add_trace(
                go.Scatter(x=list(range(len(curva_equidade))), y=curva_equidade, mode='lines', name='Capital'))
        fig.update_layout(title_text='Evolução do Capital da Estratégia', xaxis_title='Nº da Operação',
                          yaxis_title='Capital Relativo', height=350)
        return fig

    @staticmethod
    @atexit.register
    def parar_servicos():
        """Para os serviços de atualização ao sair da aplicação."""
        data_updater.parar_atualizacao()


if __name__ == "__main__":
    DashboardTrading().executar()
