import atexit
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Importações do seu projeto
from src.config.params import Params
from src.data.data_loader import DataLoader
from src.data.data_updater import data_updater
from src.models.feature_engineer import FeatureEngineer
from src.utils.risk_analyzer import RiskAnalyzer

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise Preditiva de Ativos", page_icon="📈")


class DashboardTrading:
    """Dashboard para Análise Preditiva de Ativos, com foco em clareza e metodologia."""

    def __init__(self):
        self.modelo_carregado = None
        self.ticker_selecionado = None
        data_updater.iniciar_atualizacao_automatica(tickers=Params.TICKERS)
        self._inicializar_sidebar()

    def _inicializar_sidebar(self):
        """Inicializa a barra lateral com os controles do usuário."""
        with st.sidebar:
            st.markdown("## 📈 Análise Preditiva")
            st.markdown("---")

            modelos_disponiveis = sorted([f.replace('modelo_', '').replace('.joblib', '')
                                          for f in os.listdir(Params.PATH_MODELOS) if f.endswith('.joblib')])

            if not modelos_disponiveis:
                st.warning("Nenhum modelo treinado foi encontrado.")
                st.stop()

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
        caminho = os.path.join(Params.PATH_MODELOS, f"modelo_{ticker}.joblib")
        if os.path.exists(caminho):
            try:
                return load(caminho)
            except Exception:
                return None
        return None

    def executar(self):
        """Orquestra a apresentação completa da análise."""
        if not self.ticker_selecionado:
            self._render_tela_boas_vindas()
            return

        if self.analisar_btn or self.relatorio_btn:
            self.modelo_carregado = self._carregar_modelo(self.ticker_selecionado)

            if self.modelo_carregado is None:
                st.error(f"O modelo para {self.ticker_selecionado} não foi encontrado.")
                return

            with st.spinner("Processando dados e gerando análise..."):
                _, _, X_full, y_full, precos_full, previsao = self._processar_dados_e_previsao()

            if self.relatorio_btn:
                self._render_relatorio_completo(previsao, X_full, y_full, precos_full)
            else:
                self._render_analise_em_abas(previsao, X_full, y_full, precos_full)

    def _processar_dados_e_previsao(self):
        loader = DataLoader()
        feature_engineer = FeatureEngineer()
        try:
            df_ticker, df_ibov = loader.baixar_dados_yf(self.ticker_selecionado)
        except Exception as e:
            st.warning(f"**Aviso:** Falha ao baixar dados ({e}). Usando a última versão salva no banco de dados local.")
            df_ticker = loader.carregar_do_bd(self.ticker_selecionado)
            df_ibov = pd.DataFrame()

        if df_ticker.empty:
            st.error(f"Não foi possível carregar dados para {self.ticker_selecionado}.")
            st.stop()

        X_full, y_full, precos_full, _ = feature_engineer.preparar_dataset(df_ticker, df_ibov, self.ticker_selecionado)
        previsao = self.modelo_carregado.prever_direcao(X_full.tail(1), self.ticker_selecionado)
        return df_ticker, df_ibov, X_full, y_full, precos_full, previsao

    def _render_tela_boas_vindas(self):
        st.header("Bem-vindo ao Sistema de Análise Preditiva")
        st.info("**Selecione um ativo na barra lateral e escolha um tipo de análise para começar.**", icon="👈")
        st.markdown("Esta plataforma oferece duas formas de visualizar os resultados do modelo de Machine Learning:")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Análise Interativa")
            st.write(
                "Um dashboard em abas, ideal para explorar dinamicamente os diferentes aspectos do modelo, como sua performance, os fatores de decisão e simulações de estratégia.")
        with col2:
            st.subheader("📋 Relatório Completo")
            st.write(
                "Uma visão consolidada e explicativa, perfeita para apresentação. Este relatório conta a história completa da análise, desde a previsão até a metodologia, de forma clara e didática.")

    def _render_analise_em_abas(self, previsao, X_full, y_full, precos_full):
        st.header(f"Análise Preditiva para {self.ticker_selecionado}")
        tabs = st.tabs(["🎯 **Resumo Executivo**", "🧬 **DNA do Modelo**", "📊 **Simulação de Performance**"])
        with tabs[0]:
            self._render_tab_resumo(previsao, precos_full)
        with tabs[1]:
            self._render_tab_dna(y_full)
        with tabs[2]:
            self._render_tab_simulacao(X_full, precos_full)

    def _render_relatorio_completo(self, previsao, X_full, y_full, precos_full):
        st.title(f"📋 Relatório de Análise Preditiva: {self.ticker_selecionado}")
        last_date = X_full.index[-1].strftime('%d/%m/%Y')
        next_date = (X_full.index[-1] + pd.tseries.offsets.BDay(1)).strftime('%d/%m/%Y')
        st.caption(
            f"Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} | Previsão para o pregão de {next_date}")

        # --- 1. Conclusão Executiva ---
        st.header("1. Conclusão Executiva: Qual é o Veredito?")
        recomendacao = "🟢 **OPORTUNIDADE**" if previsao['should_operate'] else "🟡 **OBSERVAR**"
        probabilidade = previsao['probabilidade']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"Para o próximo pregão, o modelo sinaliza **{recomendacao}** para o ativo **{self.ticker_selecionado}**.")
            st.metric("Confiança do Modelo na Alta", f"{probabilidade:.1%}",
                      help="Probabilidade estimada pelo modelo para um movimento de alta, conforme definido pela estratégia da Tripla Barreira.")
            st.metric("Limiar Mínimo para Operar", f"{self.modelo_carregado.threshold_operacional:.1%}",
                      help="O modelo só recomenda 'Oportunidade' se a confiança superar este valor, que foi calibrado para otimizar a performance.")
        with col2:
            self._plot_gauge_confianca(probabilidade)

        # --- 2. Diagnóstico de Confiança ---
        st.header("2. Diagnóstico de Confiança: Por que Confiar Nesta Previsão?")
        st.info(
            "A confiança na previsão não é arbitrária. Ela se baseia no desempenho histórico robusto do modelo, validado através do método **Walk-Forward**, que simula como o modelo teria performado em condições reais no passado.")
        self._render_diagnostico_confianca()

        # --- 3. Contexto de Preço ---
        st.header("3. Contexto Visual: Onde a Previsão se Encaixa?")
        st.write(
            "A análise de qualquer sinal deve ser feita no contexto do comportamento recente do preço. O gráfico abaixo mostra os últimos 90 dias do ativo e onde o sinal de oportunidade se posiciona.")
        self._plot_previsao_recente(precos_full, previsao['should_operate'])

        # --- 4. O "Cérebro" do Modelo ---
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

        # --- 5. Metodologia e Glossário ---
        st.header("5. Metodologia e Pontos de Atenção")
        self._render_glossario_metodologia()

    # --- MÉTODOS DAS ABAS ---
    def _render_tab_resumo(self, previsao: Dict[str, Any], precos_full: pd.Series):
        st.subheader("Diagnóstico e Previsão")
        col1, col2 = st.columns([2, 1])
        with col1:
            self._render_diagnostico_confianca()
        with col2:
            recomendacao = "🟢 **OPORTUNIDADE**" if previsao['should_operate'] else "🟡 **OBSERVAR**"
            st.markdown("##### Sinal para o Próximo Pregão")
            st.markdown(f"<h1>{recomendacao}</h1>", unsafe_allow_html=True)
            probabilidade = previsao['probabilidade']
            st.progress(probabilidade, text=f"{probabilidade:.1%} de Confiança na Alta")
        st.divider()
        st.subheader("Previsão no Contexto do Preço Recente")
        self._plot_previsao_recente(precos_full, previsao['should_operate'])

    def _render_tab_dna(self, y_full: pd.Series):
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
        st.warning(
            "Esta simulação é **'in-sample'** e tende a ser otimista. Serve principalmente para **ilustrar o comportamento visual** da estratégia ao longo do tempo.")
        risk_analyzer = RiskAnalyzer()
        df_sinais = self.modelo_carregado.prever_e_gerar_sinais(X_full, precos_full, self.ticker_selecionado)
        backtest_info = risk_analyzer.backtest_sinais(df_sinais)
        st.plotly_chart(self._plot_precos_sinais(df_sinais, precos_full), use_container_width=True)
        if backtest_info.get('trades', 0) > 0:
            self._exibir_metricas_backtest(backtest_info)

    # --- COMPONENTES REUTILIZÁVEIS ---
    def _render_diagnostico_confianca(self):
        st.markdown("##### Diagnóstico de Confiança do Modelo (Histórico WFV)")
        metricas = self.modelo_carregado.wfv_metrics
        score, max_score, confianca_txt, cor = self._calcular_indice_confiabilidade(metricas)
        percentual = (score / max_score) * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=percentual,
            title={'text': f"<span style='font-size:1.5em;color:{cor}'>{confianca_txt}</span>"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': cor}}))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))

        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Score de Confiança", f"{score} / {max_score}")
            st.metric("Sharpe Médio (Risco-Retorno)", f"{metricas.get('sharpe_medio', 0):.2f}")
            st.metric("Qualidade Preditiva (F1-Score)", f"{metricas.get('f1_macro_medio', 0):.2%}")

    def _render_traducao_features(self):
        with st.expander("O que esses fatores significam em termos simples? 🤔"):
            st.markdown("""
            O modelo usa indicadores técnicos para 'ler' o mercado. Aqui está uma tradução simples dos mais comuns:
            - **`rsi_14` (Índice de Força Relativa):** Mede se o ativo está "caro" (sobrecomprado) ou "barato" (sobrevendido) recentemente. Ajuda a identificar possíveis pontos de virada.
            - **`sma_ratio` (Razão de Médias Móveis):** Compara uma tendência de curto prazo com uma de longo prazo. Se a curta está acima da longa, indica uma tendência de alta.
            - **`vol_21d` (Volatilidade):** Mede o "grau de agitação" do preço. Alta volatilidade pode significar tanto risco quanto oportunidade.
            - **`momentum`:** Mede a velocidade e a força do movimento dos preços em um período.
            - **`correlacao_ibov`:** Indica se o ativo tende a se mover junto com o índice Bovespa ou na direção contrária.
            """)

    def _render_glossario_metodologia(self):
        with st.expander("Glossário: Entendendo os Termos Técnicos 📖"):
            st.markdown("""
            - **Walk-Forward Validation (WFV):** A espinha dorsal da confiança neste modelo. Em vez de testar o modelo em dados que ele já 'espiou' durante o treino, o WFV simula a passagem do tempo: o modelo treina com dados do passado (ex: 2022) e é testado em dados do 'futuro' que ele nunca viu (ex: 2023). Isso resulta em uma estimativa de performance muito mais realista e confiável.

            - **Sharpe Ratio:** A métrica mais importante para avaliar uma estratégia de investimento. Ela não mede apenas o retorno, mas o **retorno ajustado ao risco**. Um Sharpe Ratio alto (acima de 1.0 é excelente) significa que a estratégia gera bons retornos sem muita 'montanha-russa' no capital.

            - **F1-Score:** Uma métrica de Machine Learning que mede o equilíbrio entre 'acertar as oportunidades' (precisão) e 'não deixar oportunidades passarem' (recall). É mais robusta que a simples acurácia em mercados financeiros, onde os eventos de alta podem ser mais raros.

            - **Tripla Barreira:** O método usado para definir o que é um 'sucesso' ou 'fracasso'. Para cada dia, criamos três 'barreiras' no futuro (ex: 5 dias): uma de lucro (take profit), uma de perda (stop loss) e uma de tempo. O resultado da operação (alta, baixa ou neutro) é definido por qual barreira é tocada primeiro. Isso cria um alvo de previsão muito mais realista do que simplesmente 'o preço vai subir ou cair amanhã?'.
            """)
        st.warning(
            "⚠️ **Aviso Legal:** Esta é uma ferramenta de estudo e análise baseada em modelos estatísticos. A performance passada não é garantia de resultados futuros. Isto **não** constitui uma recomendação de investimento.")

    # --- MÉTODOS AUXILIARES E DE PLOTAGEM ---
    def _calcular_indice_confiabilidade(self, metricas: Dict[str, Any]) -> tuple[int, int, str, str]:
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
        threshold = self.modelo_carregado.threshold_operacional
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probabilidade * 100,
            title={'text': "Confiança na Alta (%)"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#007bff"},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold * 100}}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A linha vermelha indica o limiar de confiança mínimo para operar.")

    def _plot_previsao_recente(self, precos: pd.Series, sinal_positivo: bool):
        df_recente = precos.tail(90).copy().to_frame(name='Preço')
        df_recente['Tendência (20d)'] = df_recente['Preço'].rolling(20).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recente.index, y=df_recente['Preço'], mode='lines', name='Preço de Fechamento'))
        fig.add_trace(
            go.Scatter(x=df_recente.index, y=df_recente['Tendência (20d)'], mode='lines', name='Tendência (20 dias)',
                       line={'dash': 'dot', 'color': 'gray'}))
        if sinal_positivo:
            ultimo_preco = df_recente['Preço'].iloc[-1]
            proximo_dia = df_recente.index[-1] + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(x=[proximo_dia], y=[ultimo_preco], mode='markers', name='Sinal de Oportunidade',
                                     marker=dict(color='green', size=15, symbol='circle',
                                                 line={'width': 2, 'color': 'darkgreen'})))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    def _plot_matriz_confusao(self, y_full: pd.Series):
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

    def _forcar_download_dados(self):
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

    def _plot_precos_sinais(self, df_sinais, precos):
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

    def _exibir_metricas_backtest(self, metricas):
        cols = st.columns(4)
        cols[0].metric("Retorno Total", f"{metricas.get('retorno_total', 0):.2%}")
        cols[1].metric("Sharpe Ratio", f"{metricas.get('sharpe', 0):.2f}")
        cols[2].metric("Max Drawdown", f"{metricas.get('max_drawdown', 0):.2%}")
        cols[3].metric("Taxa de Acerto", f"{metricas.get('win_rate', 0):.2%}")

    @staticmethod
    @atexit.register
    def parar_servicos():
        data_updater.parar_atualizacao()


if __name__ == "__main__":
    DashboardTrading().executar()