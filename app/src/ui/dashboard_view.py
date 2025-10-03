from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.backtesting.risk_analyzer import RiskAnalyzer
from src.config.params import Params


class DashboardView:
    """Responsável por toda a lógica de renderização do dashboard no Streamlit."""

    def __init__(self, st_component):
        """Inicializa a View com o componente Streamlit."""
        self.st = st_component

    def render_tela_boas_vindas(self):
        """Renderiza a tela inicial de boas-vindas com informações sobre o projeto."""
        self.st.title("Bem-vindo ao Sistema de Análise Preditiva")
        self.st.markdown("---")
        self.st.header("Entendendo o Modelo: Um Guia Rápido")
        self.st.info("**Selecione um ativo na barra lateral para começar.**", icon="👈")
        col1, col2 = self.st.columns(2)
        with col1:
            self.st.subheader("🎯 O Que é Este Projeto?")
            self.st.write(
                "Este é um sistema de **apoio à decisão** baseado em Machine Learning. "
                "Seu objetivo **não é** dar recomendações de compra ou venda, mas sim **identificar oportunidades potenciais** "
                "de alta para um ativo, com base em padrões históricos.")
            self.st.subheader("🧠 Como o Modelo 'Pensa'?")
            self.st.write(
                "Utilizamos um modelo de **Classificação** que prevê a **direção** do movimento. O alvo é definido pela "
                "**Metodologia da Tripla Barreira**, que considera uma janela de tempo futura para determinar se uma operação "
                "teria sido um sucesso (lucro), um fracasso (stop) ou neutra.")
        with col2:
            self.st.subheader("🔬 Como a Confiança é Medida?")
            self.st.write(
                "A performance do modelo é avaliada pelo método de **Validação Walk-Forward (WFV)**. Este processo simula "
                "a operação em tempo real, representando uma estimativa muito mais honesta de seu desempenho.")
            self.st.subheader("⚠️ Limitações e Boas Práticas")
            self.st.write(
                "- **Performance Passada Não Garante Futuro:** O mercado é dinâmico e padrões podem não se repetir.\n"
                "- **Não é uma Bola de Cristal:** Fatores macroeconômicos e notícias não estão no escopo do modelo.\n"
                "- **Use como Ferramenta:** Esta análise deve ser usada como mais uma camada de informação em seu processo de decisão.")

    def render_main_layout(self, ticker, modelo, dados, validacao_recente, metricas_validacao, data_treinamento=None):
        """Renderiza o layout principal, incluindo o painel de veredito e as abas de análise profunda."""

        data_base = dados.get("data_base_analise")
        data_alvo = dados.get("data_alvo_previsao")

        texto_data = ""
        if data_base and data_alvo:
            texto_data = (f"Análise baseada nos dados de **{data_base.strftime('%d/%m/%Y')}**. "
                          f"Previsão para o próximo dia útil: **{data_alvo.strftime('%d/%m/%Y')}**.")

        self.st.header(f"Análise Preditiva para {ticker}")
        if texto_data:
            self.st.markdown(texto_data)

        if data_treinamento:
            self.st.caption(f"🗓️ &nbsp; Modelo treinado em: {data_treinamento.strftime('%d/%m/%Y às %H:%M')}")

        self._render_verdict_panel(modelo, dados)

        self.st.info("Para entender os detalhes por trás do veredito, explore as abas de análise abaixo.", icon="👇")

        tab1, tab2, tab3 = self.st.tabs([
            "🔎 **Diagnóstico da Previsão**",
            "📈 **Performance Histórica**",
            "📊 **Análise de Mercado**"
        ])

        with tab1:
            self._render_tab_diagnostico_previsao(modelo, dados, validacao_recente, metricas_validacao)
        with tab2:
            self._render_tab_performance_historica(modelo, dados, ticker)
        with tab3:
            self._render_tab_mercado(ticker, modelo, dados)

    def _render_verdict_panel(self, modelo, dados):
        """Renderiza o painel principal com o veredito e as métricas mais importantes."""
        previsao = dados['previsao']
        wfv_metrics = modelo.wfv_metrics
        score, max_score, _, _ = self._calcular_indice_confiabilidade(wfv_metrics)

        recomendacao = "🟢 **OPORTUNIDADE**" if previsao['should_operate'] else "🟡 **OBSERVAR**"

        st.markdown("---")
        cols = self.st.columns(5)

        with cols[0]:
            st.markdown(f"<h3 style='text-align: center;'>Veredito</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{recomendacao}</h2>", unsafe_allow_html=True)

        cols[1].metric("Probabilidade de Alta", f"{previsao['probabilidade']:.1%}",
                       help="Probabilidade estimada pelo modelo para a ocorrência de um evento de alta, conforme a metodologia da Tripla Barreira.")
        cols[2].metric("Score de Robustez", f"{score}/{max_score}",
                       help="Pontuação de 0 a 9 que resume a robustez histórica do modelo para este ativo, com base na validação Walk-Forward. Scores mais altos indicam maior confiança histórica.")
        cols[3].metric("Sharpe Médio (WFV)", f"{wfv_metrics.get('sharpe_medio', 0):.2f}",
                       help="Mede o retorno ajustado ao risco na validação mais robusta (Walk-Forward). Acima de 0.5 é bom; acima de 1.0 é excelente.")

        wfv_performance = modelo.gerar_performance_wfv_agregada(dados['y_full'], dados['precos_full'], dados['t1'])
        cols[4].metric("Taxa de Acerto (WFV)", f"{wfv_performance.get('win_rate', 0):.1%}",
                       help="Percentual de operações que resultaram em lucro durante a validação Walk-Forward. Um valor consistentemente acima de 50% é desejável.")
        st.markdown("---")

    def _render_tab_diagnostico_previsao(self, modelo, dados, validacao_recente, metricas_validacao):
        """Renderiza a aba focada em explicar e validar a previsão atual."""
        col1, col2 = self.st.columns(2)
        with col1:
            self.st.subheader("Explicabilidade da Previsão (SHAP)")
            self.st.caption("Quais fatores mais influenciaram a recomendação de hoje?")
            self._render_secao_previsao_shap(dados['X_full'], modelo)

        with col2:
            self.st.subheader("Saúde do Modelo (Data Drift)")
            self.st.caption("O mercado atual se parece com os dados de treino?")
            self._render_secao_saude_modelo(dados['X_full'], modelo)

        self.st.divider()
        self.st.subheader(f"Validação de Performance (Últimos {Params.UI_VALIDATION_DAYS} Sinais)")
        self.st.caption("Como o modelo performou nos últimos dias?")
        self._render_secao_validacao_recente(validacao_recente, metricas_validacao)

    def _render_tab_performance_historica(self, modelo, dados, ticker):
        """Renderiza a aba que consolida todas as análises de performance histórica."""

        self.st.subheader("Resultados da Validação Walk-Forward (Out-of-Sample)")
        self.st.success(
            "Esta é a estimativa mais **realista e confiável** da performance do modelo, pois simula operações em dados que o modelo nunca viu durante o treino.")

        self._render_explicacao_score_confianca()

        wfv_performance = modelo.gerar_performance_wfv_agregada(dados['y_full'], dados['precos_full'], dados['t1'])

        if wfv_performance['trades'] > 0:
            cols = self.st.columns(4)
            cols[0].metric("Retorno Total (WFV)", f"{wfv_performance['retorno_total']:.2%}", help="Retorno percentual acumulado ao final da validação Walk-Forward. Representa o ganho/perda total da estratégia na simulação mais realista.")
            cols[1].metric("Nº de Trades (WFV)", f"{wfv_performance['trades']}", help="Número total de operações executadas na validação WFV. Um número maior de trades (>50) confere maior robustez estatística aos resultados.")
            cols[2].metric("Taxa de Acerto (WFV)", f"{wfv_performance['win_rate']:.1%}", help="Percentual de operações que resultaram em lucro durante a validação Walk-Forward. Um valor consistentemente acima de 50% é desejável.")
            cols[3].metric("Max Drawdown (WFV)", f"{wfv_performance['max_drawdown']:.2%}",
                           help="A maior queda percentual do capital a partir de um pico durante a simulação WFV. Mede o pior cenário de perda histórica. Abaixo de -20% exige atenção.")

            col_chart1, col_chart2 = self.st.columns(2)
            with col_chart1:
                self._plot_wfv_equity_curve(wfv_performance)
            with col_chart2:
                self._plot_rolling_sharpe_wfv(wfv_performance)
        else:
            self.st.warning("Não há trades suficientes na validação WFV para gerar a curva de capital.")

        self.st.divider()
        self.st.subheader("Análise Visual dos Sinais WFV")
        self.st.caption(
            "Este gráfico mostra exatamente onde os sinais de oportunidade teriam ocorrido em um cenário realista de teste.")
        self._plot_wfv_signals_on_price(dados['precos_full'], wfv_performance)

        self.st.divider()

        self.st.subheader("Simulação Otimista (Todos os Dados)")
        self.st.warning(
            "Esta simulação utiliza **todos os dados disponíveis (incluindo os de treino)** e tende a ser uma visão **otimista** da performance. Serve principalmente para ilustrar o comportamento geral da estratégia.")
        self._render_secao_simulacao_completa(ticker, modelo, dados)

    def _render_tab_mercado(self, ticker, modelo, dados):
        """Renderiza a aba de Análise de Mercado."""
        self.st.subheader("Previsão no Contexto do Preço Recente")
        self._plot_previsao_recente(dados['precos_full'], dados['previsao']['should_operate'])

        self.st.divider()

        self.st.subheader(f"Performance Relativa: {ticker} vs. IBOVESPA")
        self._plot_performance_vs_ibov(dados['precos_full'], dados['df_ibov'], ticker)

        self.st.divider()
        self.st.subheader("DNA do Modelo: Fatores e Performance de Classificação")
        self.st.caption("Esta seção aprofunda no funcionamento interno do modelo, revelando quais fatores são mais importantes para sua decisão e quão bem ele consegue distinguir entre os diferentes cenários de mercado (alta, baixa e neutro).")
        col1, col2 = self.st.columns(2)
        with col1:
            self.st.markdown("**Variáveis Mais Influentes**")
            self._plot_importancia_features(modelo)
        with col2:
            self.st.markdown("**Previsão vs. Realidade (Classificação)**")
            self._plot_matriz_confusao(dados['y_full'], modelo)

        self._render_traducao_features()
        self._render_glossario_metodologia()

    def _render_secao_simulacao_completa(self, ticker, modelo, dados):
        risk_analyzer = RiskAnalyzer()
        df_sinais = modelo.prever_e_gerar_sinais(dados['X_full'], dados['precos_full'], ticker)
        backtest_info = risk_analyzer.backtest_sinais(df_sinais)

        self.st.plotly_chart(self._plot_precos_sinais(df_sinais, dados['precos_full']), use_container_width=True)
        self.st.caption("Como interpretar: Este gráfico mostra o histórico de preços do ativo com marcadores verdes nos dias em que o modelo teria sinalizado uma 'Oportunidade', com base em todos os dados (visão otimista).")


        if backtest_info.get('trades', 0) > 0:
            self._render_secao_metricas_simulacao(backtest_info)
            self._render_secao_risco_capital(backtest_info)
            self._render_secao_sensibilidade(dados['X_full'], dados['precos_full'], ticker, modelo)
        else:
            self.st.info("A simulação não gerou nenhuma operação para este ativo.")

    def _render_secao_validacao_recente(self, resultados_validacao: list, metricas: dict):
        if not resultados_validacao:
            self.st.warning("Não há dados suficientes para gerar a validação recente.")
            return

        num_oportunidades = metricas.get('num_oportunidades_recente', 0)

        if num_oportunidades > 0:
            col1, col2, col3 = self.st.columns(3)
            col1.metric("Taxa de Acerto (Oportunidades)", f"{metricas.get('taxa_acerto', 0):.1%}",
                         help="Dos sinais de 'OPORTUNIDADE' gerados, qual % foi seguido por uma variação diária positiva.")
            col2.metric("Assertividade Geral", f"{metricas.get('assertividade_geral', 0):.1%}",
                        help="Percentual de dias em que o modelo tomou a decisão correta (acertando altas ou evitando perdas).")
            col3.metric("Retorno Médio Diário (nos Acertos)", f"{metricas.get('retorno_medio_acertos', 0):.2%}",
                         help="A variação média do preço no dia seguinte para os sinais de 'OPORTUNIDADE' que o modelo acertou.")
        else:
            col1, col2 = self.st.columns([1, 3])
            with col1:
                self.st.metric("Assertividade Geral", f"{metricas.get('assertividade_geral', 0):.1%}",
                                 help="Percentual de dias em que o modelo tomou a decisão correta (neste caso, o quão bem ele evitou perdas).")
            with col2:
                self.st.info(
                    "O modelo não identificou sinais de 'Oportunidade' no período recente. A 'Assertividade Geral' reflete o quão bem ele evitou perdas nos dias em que recomendou observar.",
                    icon="ℹ️")

        df_validacao = pd.DataFrame(resultados_validacao)

        def formatar_resultado_real(label):
            if label == 1: return "📈 ALTA"
            if label == -1: return "📉 BAIXA"
            if label == 0: return "↔️ NEUTRO"
            if label == "⏳ Aguardando": return "⏳ Aguardando"
            return "N/A"

        def estilo_performance(val):
            cor = ""
            if "Acerto" in val or "Evitou Perda" in val:
                cor = "#28a745"
            elif "Erro" in val:
                cor = "#dc3545"
            elif "Acompanhamento" in val:
                cor = "#007bff"
            return f'color: {cor}; font-weight: bold;'

        def estilo_variacao(val):
            if pd.isna(val): return ''
            cor = "#28a745" if val > 0 else ("#dc3545" if val < 0 else "")
            return f'color: {cor};'

        df_validacao['Resultado Real (Tripla Barreira)'] = df_validacao['Resultado Real (Tripla Barreira)'].apply(
            formatar_resultado_real)

        self.st.dataframe(
            df_validacao.style
            .format({"Probabilidade de Alta": "{:.1%}", "Variação Diária Real": "{:+.2%}"})
            .map(estilo_performance, subset=['Performance'])
            .map(estilo_variacao, subset=['Variação Diária Real']),
            use_container_width=True, hide_index=True)

    def _render_explicacao_score_confianca(self):
        """Renderiza um expander explicando a metodologia do Score de Confiança."""
        with self.st.expander("Como o Score de Robustez é calculado? 🤔"):
            self.st.markdown("""
            O score é uma nota de 0 a 9 que resume a robustez histórica do modelo para este ativo, com base em 3 pilares da validação Walk-Forward (WFV):

            **1. Risco-Retorno (Sharpe Ratio):** Mede a qualidade do retorno em relação ao risco assumido.
            - `Sharpe > 1.0`: **+3 pontos** (Excelente)
            - `Sharpe > 0.3`: **+2 pontos** (Bom)
            - `Sharpe > -0.1`: **+1 ponto** (Aceitável)

            **2. Qualidade Preditiva (F1-Score):** Mede o equilíbrio do modelo em acertar as oportunidades sem gerar muitos alarmes falsos.
            - `F1-Score > 65%`: **+3 pontos** (Ótima)
            - `F1-Score > 55%`: **+2 pontos** (Boa)
            - `F1-Score > 50%`: **+1 ponto** (Razoável)

            **3. Relevância Estatística (Média de Trades):** Garante que a performance foi observada em um número mínimo de operações.
            - `Média > 8 trades/fold`: **+3 pontos** (Ativo)
            - `Média > 4 trades/fold`: **+2 pontos** (Moderado)
            - `Média > 2.5 trades/fold`: **+1 ponto** (Seletivo)

            ---
            **Score Final:**
            - **7-9 Pontos:** <span style='color:green;font-weight:bold;'>Alta Confiança</span>
            - **4-6 Pontos:** <span style='color:orange;font-weight:bold;'>Média Confiança</span>
            - **0-3 Pontos:** <span style='color:red;font-weight:bold;'>Baixa Confiança</span>
            """, unsafe_allow_html=True)

    def _plot_rolling_sharpe_wfv(self, performance_data: dict, janela: int = 25):
        """Plota o Sharpe Ratio móvel da validação Walk-Forward."""
        retornos = performance_data.get('retornos', [])
        if len(retornos) < janela:
            self.st.caption("Não há trades suficientes para calcular a performance móvel.")
            return

        def sharpe_anualizado(r):
            if r.std() == 0: return 0.0
            return (r.mean() / r.std()) * np.sqrt(Params.DIAS_UTEIS_ANUAIS)

        rolling_sharpe = pd.Series(retornos).rolling(window=janela).apply(sharpe_anualizado, raw=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe, mode='lines',
            name='Sharpe Móvel', line=dict(color='purple')
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Bom")
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Ótimo")

        fig.update_layout(
            title=f'Sharpe Ratio Móvel (Janela de {janela} trades)',
            xaxis_title='Nº de Operações (ordem cronológica)',
            yaxis_title='Sharpe Ratio Anualizado',
            height=400
        )
        self.st.plotly_chart(fig, use_container_width=True)
        self.st.caption("Como interpretar: Este gráfico mostra a evolução do retorno ajustado ao risco. Uma linha que se mantém consistentemente acima de 0.5 (bom) ou 1.0 (ótimo) indica uma performance robusta e estável ao longo do tempo.")


    def _plot_wfv_equity_curve(self, performance_data: dict):
        curva_capital = performance_data.get('equity_curve', [])
        fig = go.Figure()
        if len(curva_capital) > 1:
            fig.add_trace(
                go.Scatter(x=list(range(len(curva_capital))), y=curva_capital, mode='lines', name='Capital (WFV)'))
        fig.update_layout(
            title_text='Evolução do Capital (Walk-Forward)',
            xaxis_title='Nº de Operações (ordem cronológica)',
            yaxis_title='Capital Relativo (Início = R$1)',
            height=400)
        self.st.plotly_chart(fig, use_container_width=True)
        self.st.caption("Como interpretar: Uma curva consistentemente ascendente indica uma estratégia lucrativa. A inclinação representa a taxa de retorno. Períodos planos indicam que o modelo não encontrou oportunidades.")


    def _render_secao_previsao_shap(self, X_full, modelo):
        if not hasattr(modelo, 'shap_explainer') or modelo.shap_explainer is None:
            self.st.warning("O explainer SHAP não foi encontrado neste modelo.")
            return
        with self.st.spinner("Calculando valores SHAP..."):
            X_last = X_full[modelo.features_selecionadas].tail(1)
            X_last_scaled = modelo.scaler.transform(X_last)
            shap_explanation_raw = modelo.shap_explainer(X_last_scaled)
            idx_classe_1 = np.where(modelo.label_encoder.classes_ == 1)[0][0]
            shap_explanation_for_plot = shap_explanation_raw[0, :, idx_classe_1]
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_explanation_for_plot, max_display=15, show=False)
            plt.tight_layout()
            self.st.pyplot(fig)
            plt.close()
            self.st.caption("""
                **Como ler este gráfico:** O valor `f(x)` final representa a "Probabilidade de Alta".
                As **barras azuis** são os fatores que empurraram a probabilidade para cima (contribuíram para um sinal de OPORTUNIDADE).
                As **barras vermelhas** empurraram a probabilidade para baixo.
                        """)

    def _render_secao_saude_modelo(self, X_full, modelo):
        if not hasattr(modelo, 'training_data_profile') or modelo.training_data_profile is None:
            self.st.warning("O perfil dos dados de treino não foi encontrado.")
            return
        baseline_profile = pd.DataFrame(modelo.training_data_profile)
        X_recent = X_full[modelo.features_selecionadas].tail(60)
        X_recent_scaled = pd.DataFrame(modelo.scaler.transform(X_recent), columns=X_recent.columns)

        key_features = Params.UI_DRIFT_KEY_FEATURES
        cols = self.st.columns(len(key_features))
        for i, feature in enumerate(key_features):
            if feature in baseline_profile.columns and feature in X_recent_scaled.columns:
                with cols[i]:
                    self._plot_drift_distribution(baseline_profile[feature], X_recent_scaled[feature], feature)

    def _plot_drift_distribution(self, baseline_series, current_series, feature_name):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=baseline_series, name='Treino', opacity=0.75, histnorm='probability density'))
        fig.add_trace(go.Histogram(x=current_series, name='Recente', opacity=0.75, histnorm='probability density'))
        fig.update_layout(barmode='overlay', title_text=f'Drift de {feature_name}', height=250,
                          margin=dict(t=30, b=10, l=10, r=10))
        _, p_value = ks_2samp(baseline_series.dropna(), current_series.dropna())
        self.st.plotly_chart(fig, use_container_width=True)
        if p_value < 0.05:
            self.st.error(f"P-valor: {p_value:.3f}. Drift detectado!")
        else:
            self.st.success(f"P-valor: {p_value:.3f}. Sem drift.")

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
            proximo_dia = df_recente.index[-1] + pd.tseries.offsets.BDay(1)
            fig.add_trace(go.Scatter(x=[proximo_dia], y=[ultimo_preco], mode='markers', name='Sinal de Oportunidade',
                                      marker=dict(color='green', size=15, symbol='circle',
                                                 line={'width': 2, 'color': 'darkgreen'})))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        self.st.plotly_chart(fig, use_container_width=True)

    def _plot_wfv_signals_on_price(self, precos_full: pd.Series, performance_data: dict):
        """Plota os sinais da validação WFV (out-of-sample) sobre o gráfico de preços."""
        sinais_wfv = performance_data.get('sinais_wfv', [])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precos_full.index, y=precos_full, mode='lines', name='Preço de Fechamento'))

        if sinais_wfv:
            df_sinais = pd.DataFrame(sinais_wfv)
            fig.add_trace(go.Scatter(
                x=df_sinais['data'],
                y=df_sinais['preco'],
                mode='markers',
                name='Sinal de Oportunidade (WFV)',
                marker=dict(color='limegreen', size=8, symbol='triangle-up', line={'width': 1, 'color': 'darkgreen'})
            ))

        fig.update_layout(
            title='Preço Histórico com Sinais Out-of-Sample (Validação Walk-Forward)',
            xaxis_title='Data',
            yaxis_title='Preço',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        self.st.plotly_chart(fig, use_container_width=True)
        self.st.caption("Como interpretar: Os triângulos verdes mostram os pontos exatos onde o modelo teria identificado uma oportunidade de entrada durante a simulação mais realista (Walk-Forward). Isso ajuda a visualizar o comportamento do modelo em diferentes condições de mercado.")

    def _plot_performance_vs_ibov(self, precos_ativo: pd.Series, df_ibov: pd.DataFrame, ticker: str):
        if df_ibov is None or df_ibov.empty:
            self.st.warning("Não foi possível carregar os dados do IBOVESPA para comparação.")
            return
        ibov_close = df_ibov.get('Close_IBOV', df_ibov.get('Close'))
        if ibov_close is None:
            self.st.error("Coluna de fechamento do IBOVESPA não encontrada.")
            return
        df_comp = pd.DataFrame(precos_ativo).rename(columns={'Close': 'Ativo'})
        df_comp['IBOV'] = ibov_close
        df_comp = df_comp.dropna().tail(252)
        if df_comp.empty or len(df_comp) < 2:
            self.st.warning("Não há dados suficientes para a comparação entre o ativo e o IBOVESPA.")
            return
        df_normalizado = (df_comp / df_comp.iloc[0]) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_normalizado.index, y=df_normalizado['Ativo'], mode='lines', name=ticker))
        fig.add_trace(go.Scatter(x=df_normalizado.index, y=df_normalizado['IBOV'], mode='lines', name='IBOVESPA',
                                 line={'dash': 'dot', 'color': 'gray'}))
        fig.update_layout(title_text='Performance Normalizada (Base 100)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        self.st.plotly_chart(fig, use_container_width=True)

    def _plot_importancia_features(self, modelo: Any):
        try:
            importances = modelo.modelo_final.feature_importances_
            features = modelo.features_selecionadas
            df_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance',
                                                                                                ascending=False).head(
                10).sort_values('importance', ascending=True)
            fig = go.Figure(go.Bar(x=df_imp['importance'], y=df_imp['feature'], orientation='h'))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            self.st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            self.st.warning(f"Não foi possível gerar o gráfico de importância: {e}")

    def _plot_matriz_confusao(self, y_full: pd.Series, modelo: Any):
        try:
            from src.models.validation import PurgedKFoldCV

            if not hasattr(modelo, 'cv_gen') or not hasattr(modelo.cv_gen, 't1'):
                self.st.warning("Não foi possível gerar a Matriz de Confusão: objeto 't1' não encontrado no modelo.")
                return

            common_index = modelo.X_scaled.index.intersection(y_full.index).intersection(modelo.cv_gen.t1.index)

            x_aligned = modelo.X_scaled.loc[common_index]
            y_aligned = y_full.loc[common_index]
            t1_aligned = modelo.cv_gen.t1.loc[common_index]


            cv_gen_temp = PurgedKFoldCV(n_splits=modelo.cv_gen.n_splits, t1=t1_aligned,
                                        purge_days=modelo.cv_gen.purge_days)

            _, test_idx = list(cv_gen_temp.split(x_aligned))[-1]

            y_encoded_aligned = modelo.label_encoder.transform(y_aligned)
            y_test_encoded = y_encoded_aligned[test_idx]
            y_test_labels = modelo.label_encoder.inverse_transform(y_test_encoded)

            preds_labels = modelo.label_encoder.inverse_transform(
                modelo.modelo_final.predict(x_aligned.iloc[test_idx]))

            cm = confusion_matrix(y_test_labels, preds_labels, labels=[-1, 0, 1])
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BAIXA', 'NEUTRO', 'ALTA']).plot(ax=ax,
                                                                                                         cmap='Blues',
                                                                                                         values_format='d')
            ax.set_title("Previsões vs. Realidade")
            self.st.pyplot(fig)
            self.st.caption("Matriz de confusão do último 'fold' da validação.")
        except Exception as e:
            self.st.warning(f"Não foi possível gerar a Matriz de Confusão: {e}")

    def _render_traducao_features(self):
        """ Renderiza a seção que traduz os fatores técnicos em termos simples."""
        with self.st.expander("O que esses fatores significam em termos simples? 🤔"):
            self.st.markdown("""
            O modelo analisa 18 fatores (features) para tomar sua decisão. Eles são agrupados em categorias para medir diferentes aspectos do comportamento do ativo:

            ---
            #### 📈 Indicadores de Momentum (Força e Velocidade do Preço)
            * **`retorno_1d`, `retorno_3d`**: O quão forte o preço subiu ou caiu no curtíssimo prazo (últimos 1 e 3 dias).
            * **`momentum_5d`, `momentum_21d`**: Medem o "impulso" do preço nas últimas 1 e 4 semanas. Um momentum alto e positivo indica que a tendência de alta está acelerando.
            * **`rsi_14` (Índice de Força Relativa)**: Um "termômetro" de 0 a 100 que mede se o ativo está "caro" (sobrecomprado, >70) ou "barato" (sobrevendido, <30) recentemente.
            * **`stoch_14` (Estocástico)**: Similar ao RSI, mede onde o preço de fechamento está em relação à sua faixa de variação recente (máximas e mínimas). Um valor alto (>80) indica que o ativo fechou perto de sua máxima, um sinal de força.
            ---
            #### 📊 Indicadores de Tendência
            * **`sma_ratio_10_50`, `sma_ratio_50_200`**: Comparam tendências de curto prazo com as de longo prazo. Quando a razão é > 1.0, a tendência de curto prazo é mais forte, sinalizando uma possível tendência de alta (um "cruzamento dourado").
            * **`macd_hist` (Histograma MACD)**: Mede a diferença entre duas médias móveis de tendência. Quando o histograma está positivo e crescendo, indica que o momentum de alta está se fortalecendo.
            ---
            #### 🎢 Indicadores de Volatilidade (Agitação do Mercado)
            * **`vol_21d`**: A medida estatística clássica da volatilidade. Informa o quão "nervoso" ou instável o preço do ativo tem sido no último mês.
            * **`vol_of_vol_10d` (Volatilidade da Volatilidade)**: Mede se a própria volatilidade está estável ou mudando rapidamente. Uma alta neste indicador pode sinalizar uma mudança no comportamento do mercado.
            * **`atr_14_norm` (ATR Normalizado)**: Mede o "tamanho médio do candle" dos últimos 14 dias, normalizado pelo preço. É uma medida pura do range de negociação diário.
            * **`bollinger_pct` (%B)**: Indica onde o preço atual está em relação às Bandas de Bollinger. Um valor > 1.0 significa que o preço fechou acima da banda superior (movimento forte, talvez sobrecomprado). Um valor < 0 significa que fechou abaixo da banda inferior (movimento fraco, talvez sobrevendido).
            ---
            #### 📉 Indicadores de Volume (Intensidade da Negociação)
            * **`volume_ratio_21d`**: Compara o volume de negociação de hoje com a média do último mês. Um valor > 1.0 indica um interesse excepcionalmente alto no ativo, o que pode validar um movimento de preço.
            * **`obv_norm_21d` (On-Balance Volume Normalizado)**: É um total acumulado do volume, que aumenta em dias de alta e diminui em dias de baixa. Mede a pressão de compra e venda acumulada.
            * **`cmf_20` (Chaikin Money Flow)**: Mede o fluxo de dinheiro para dentro ou para fora do ativo. Um valor positivo indica pressão compradora, enquanto um negativo indica pressão vendedora.
            ---
            #### 🌍 Indicadores de Mercado (Contexto)
            * **`correlacao_ibov_20d`**: Mede o quão "em sintonia" o ativo está com o índice Ibovespa. Um valor próximo de +1 indica que ele tende a seguir o mercado; próximo de -1, que ele se move na direção oposta.
            * **`ibov_acima_sma50`**: Um simples sinal (Sim/Não) que verifica se o mercado em geral (Ibovespa) está em uma tendência de médio prazo de alta. Um sinal de oportunidade no ativo pode ser mais forte se o mercado como um todo também estiver subindo.
            """)

    def _render_glossario_metodologia(self):
        """ Renderiza a seção de glossário e metodologia com explicações dos termos técnicos usados."""
        with self.st.expander("Glossário: Entendendo os Termos Técnicos 📖"):
            self.st.markdown("""
            - **Walk-Forward Validation (WFV):** A espinha dorsal da confiança neste modelo. Em vez de testar o modelo em dados que ele já 'espiou' durante o treino, o WFV simula a passagem do tempo: o modelo treina com dados do passado (ex: 2022) e é testado em dados do 'futuro' que ele nunca viu (ex: 2023). Isso resulta em uma estimativa de performance muito mais realista e confiável.
            - **Sharpe Ratio:** A métrica mais importante para avaliar uma estratégia de investimento. Ela não mede apenas o retorno, mas o **retorno ajustado ao risco**. Um Sharpe Ratio alto (acima de 1.0 é excelente) significa que a estratégia gera bons retornos sem muita 'montanha-russa' no capital.
            - **F1-Score:** Uma métrica de Machine Learning que mede o equilíbrio entre 'acertar as oportunidades' (precisão) e 'não deixar oportunidades passarem' (recall). É mais robusta que a simples acurácia em mercados financeiros, onde os eventos de alta podem ser mais raros.
            - **Tripla Barreira:** O método usado para definir o que é um 'sucesso' ou 'fracasso'. Para cada dia, criamos três 'barreiras' no futuro (ex: 5 dias): uma de lucro (take profit), uma de perda (stop loss) e uma de tempo. O resultado da operação (alta, baixa ou neutro) é definido por qual barreira é tocada primeiro. Isso cria um alvo de previsão muito mais realista do que simplesmente 'o preço vai subir ou cair amanhã?'.
            """)
        self.st.warning(
            "⚠️ **Aviso Legal:** Esta é uma ferramenta de estudo e análise baseada em modelos estatísticos. A performance passada não é garantia de resultados futuros. Isto **não** constitui uma recomendação de investimento.")

    @staticmethod
    def _calcular_indice_confiabilidade(metricas: Dict[str, Any]) -> tuple[int, int, str, str]:
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
        if score >= 7: return score, max_score, "Alta Robustez", "green"
        if score >= 4: return score, max_score, "Média Robustez", "orange"
        return score, max_score, "Baixa Robustez", "red"

    def _exibir_metricas_backtest(self, metricas: Dict[str, Any]):
        self.st.subheader("Métricas de Performance da Simulação In-Sample")
        cols = self.st.columns(4)
        cols[0].metric("Retorno Total", f"{metricas.get('retorno_total', 0):.2%}", help="O retorno percentual acumulado ao final de toda a simulação, assumindo que os lucros são reinvestidos.")
        cols[1].metric("Sharpe Ratio", f"{metricas.get('sharpe', 0):.2f}", help="Mede o retorno ajustado ao risco. Acima de 1.0 é considerado excelente, pois indica que os retornos superam a volatilidade de forma consistente.")
        cols[2].metric("Sortino Ratio", f"{metricas.get('sortino', 0):.2f}", help="Similar ao Sharpe, mas foca apenas no risco de perdas (volatilidade negativa). Um valor acima de 2.0 é excelente, pois indica boa proteção contra perdas.")
        cols[3].metric("Nº de Trades", f"{metricas.get('trades', 0)}", help="Número total de operações de compra e venda executadas. Um número maior de trades (>50) confere maior robustez estatística aos resultados.")
        col_q1, col_q2, col_q3, col_q4 = self.st.columns(4)
        col_q1.metric("Taxa de Acerto", f"{metricas.get('win_rate', 0):.2%}", help="Percentual de operações que resultaram em lucro. Um valor consistentemente acima de 50% é desejável.")
        col_q2.metric("Profit Factor", f"{metricas.get('profit_factor', 0):.2f}",
                      help="Quanto o sistema ganhou para cada R$ 1 que perdeu. Ex: um fator de 2.0 significa que os lucros totais foram o dobro das perdas totais. Acima de 1.5 é bom.")
        col_q3.metric("Payoff Ratio", f"{metricas.get('payoff_ratio', 0):.2f}",
                      help="Compara o tamanho da operação vencedora média com a perdedora média. Um payoff de 2.0 significa que, em média, um trade de ganho foi 2x maior que um trade de perda. Acima de 1.5 é considerado bom.")
        col_q4.metric("Max Drawdown", f"{metricas.get('max_drawdown', 0):.2%}",
                      help="A maior queda percentual do capital a partir de um pico. Mede o pior cenário de perda histórica. Abaixo de -20% exige atenção.")

    def _render_secao_risco_capital(self, backtest_info):
        self.st.subheader("Análise de Risco e Capital (In-Sample)")
        self.st.caption("Esta seção analisa visualmente a jornada do capital e os períodos de perdas ao longo da simulação completa.")
        col1, col2 = self.st.columns(2)
        with col1:
            self.st.plotly_chart(self._plot_equidade(backtest_info), use_container_width=True)
            self.st.caption("Interpretação: A curva de equidade mostra a evolução do capital ao longo das operações. O ideal é uma tendência de alta constante com poucas quedas (drawdowns).")
        with col2:
            self._plot_drawdown_curve(backtest_info)


    @staticmethod
    def _plot_precos_sinais(df_sinais, precos):
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
    def _plot_equidade(backtest_info: Dict[str, Any]) -> go.Figure:
        curva_equidade = backtest_info.get('equity_curve', [])
        fig = go.Figure()
        if len(curva_equidade) > 1:
            fig.add_trace(
                go.Scatter(x=list(range(len(curva_equidade))), y=curva_equidade, mode='lines', name='Capital'))
        fig.update_layout(title_text='Evolução do Capital (Curva de Equidade)', xaxis_title='Nº da Operação',
                          yaxis_title='Capital Relativo', height=350)
        return fig

    def _plot_drawdown_curve(self, backtest_info: Dict[str, Any]):
        drawdown_series = backtest_info.get('drawdown_series', [])
        if not drawdown_series: return
        df_dd = pd.DataFrame(drawdown_series, columns=['Drawdown'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_dd.index, y=df_dd['Drawdown'], fill='tozeroy', name='Drawdown', line_color='red'))
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title_text='Curva de Drawdown (Períodos de Perda)', xaxis_title='Nº da Operação',
                          yaxis_title='Queda do Pico', height=350)
        self.st.plotly_chart(fig, use_container_width=True)
        self.st.caption("Interpretação: O gráfico de drawdown mostra os períodos de perda. O objetivo é ter 'vales' rasos e curtos, indicando que as perdas são pequenas e a recuperação é rápida.")

    def _render_secao_sensibilidade(self, X_full, precos_full, ticker, modelo):
        with self.st.expander("Análise de Sensibilidade do Threshold de Operação"):
            self.st.info(
                "Este gráfico mostra como a performance (Sharpe Ratio) da estratégia mudaria com diferentes limiares de confiança. Um pico largo em torno do threshold escolhido (linha vermelha) indica uma estratégia robusta.")
            with self.st.spinner("Analisando sensibilidade..."):
                df_sensibilidade = self._analisar_sensibilidade_threshold(X_full, precos_full, ticker, modelo)
                self._plot_sensibilidade_threshold(df_sensibilidade, modelo)

    @st.cache_data
    def _analisar_sensibilidade_threshold(_self, X: pd.DataFrame, precos: pd.Series, ticker: str,
                                          _modelo: Any) -> pd.DataFrame:
        risk_analyzer = RiskAnalyzer()
        resultados = []
        for thr in np.arange(0.40, 0.75, 0.025):
            df_sinais = _modelo.prever_e_gerar_sinais(X, precos, ticker, threshold_override=thr)
            backtest = risk_analyzer.backtest_sinais(df_sinais, verbose=False)
            resultados.append({'threshold': thr, 'sharpe': backtest.get('sharpe', 0)})
        return pd.DataFrame(resultados)

    def _plot_sensibilidade_threshold(self, df_sensibilidade, modelo):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sensibilidade['threshold'], y=df_sensibilidade['sharpe'], mode='lines+markers',
                                 name='Sharpe Ratio'))
        fig.add_vline(x=modelo.threshold_operacional, line_width=2, line_dash="dash", line_color="red",
                      annotation_text="Threshold Escolhido", annotation_position="top left")
        fig.update_layout(title="Performance (Sharpe) vs. Threshold de Decisão",
                          xaxis_title="Threshold de Decisão (Probabilidade Mínima)",
                          yaxis_title="Sharpe Ratio Anualizado")
        self.st.plotly_chart(fig, use_container_width=True)

    def _render_secao_metricas_simulacao(self, backtest_info):
        self._exibir_metricas_backtest(backtest_info)
        self.st.divider()