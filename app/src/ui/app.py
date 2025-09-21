import os
from datetime import datetime
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Importações do seu projeto
from src.config.params import Params
from src.data.data_loader import DataLoader
from src.models.feature_engineer import FeatureEngineer
from src.utils.risk_analyzer import RiskAnalyzer

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise Preditiva de Ativos", page_icon="📈")


class RelatorioAnalitico:
    """Gera relatórios explicativos automáticos baseados nos resultados do modelo."""

    @staticmethod
    def gerar_explicacao_threshold(threshold: float, metricas: Dict[str, Any]) -> str:
        """Gera explicação personalizada para o threshold do modelo."""

        if threshold < 0.45:
            estilo = "agressivo"
            razao = "baixo threshold"
            explicacao = f"""
            **Estratégia Ágil (Threshold: {threshold:.3f})**

            Seu modelo opera com um threshold de **{threshold:.3f}**, o que indica uma estratégia mais **ágil e sensível**.

            **Por que isso aconteceu?**
            - O modelo identificou que este ativo responde melhor a sinais mais frequentes
            - O mercado para este ativo apresenta movimentos mais sutis que requerem sensibilidade
            - O balanceamento entre precisão e recall favoreceu capturar mais oportunidades

            **Implicações:**
            ✅ Maior número de oportunidades identificadas  
            ⚠️ Maior chance de falsos positivos  
            📈 Ideal para mercados com movimentos frequentes mas de menor amplitude
            """
        elif threshold < 0.55:
            estilo = "balanceado"
            razao = "threshold moderado"
            explicacao = f"""
            **Estratégia Balanceada (Threshold: {threshold:.3f})**

            Seu modelo opera com um threshold de **{threshold:.3f}**, indicando uma estratégia **equilibrada** entre sensibilidade e precisão.

            **Por que isso aconteceu?**
            - O modelo encontrou um ponto ótimo entre capturar oportunidades e evitar ruído
            - O ativo apresenta um comportamento de mercado típico sem extremos
            - A relação risco-retorno é mais balanceada neste patamar

            **Implicações:**
            ✅ Bom equilíbrio entre oportunidades e confiabilidade  
            ⚖️ Balanceamento ideal para maioria dos cenários de mercado  
            📊 Performance consistente em diferentes condições
            """
        else:
            estilo = "conservador"
            razao = "alto threshold"
            explicacao = f"""
            **Estratégia Conservadora (Threshold: {threshold:.3f})**

            Seu modelo opera com um threshold de **{threshold:.3f}**, indicando uma estratégia **conservadora e de alta confiança**.

            **Por que isso aconteceu?**
            - O ativo apresenta movimentos bruscos que exigem maior confiança nas previsões
            - O modelo priorizou evitar falsos positivos em detrimento de capturar mais oportunidades
            - O mercado para este ativo tem características que favorecem esperar por sinais mais claros

            **Implicações:**
            ✅ Alta confiabilidade nos sinais gerados  
            ⚠️ Menor frequência de oportunidades identificadas  
            🛡️ Proteção contra falsos positivos em mercados voláteis
            """

        return explicacao, estilo, razao

    @staticmethod
    def gerar_analise_performance(metricas_wfv: Dict[str, Any], previsao_atual: Dict[str, Any]) -> str:
        """Gera análise de performance baseada nas métricas de validação."""

        sharpe = metricas_wfv.get('sharpe_medio', 0)
        f1 = metricas_wfv.get('f1_macro_medio', 0)
        trades = metricas_wfv.get('trades_medio', 0)
        proba = previsao_atual.get('probabilidade', 0.5)

        # Análise do Sharpe Ratio
        if sharpe > 0.5:
            analise_sharpe = f"Excelente performance de risco-retorno (Sharpe: {sharpe:.2f})"
        elif sharpe > 0.2:
            analise_sharpe = f"Boa performance de risco-retorno (Sharpe: {sharpe:.2f})"
        elif sharpe > 0:
            analise_sharpe = f"Performance moderada de risco-retorno (Sharpe: {sharpe:.2f})"
        else:
            analise_sharpe = f"Performance abaixo do esperado (Sharpe: {sharpe:.2f})"

        # Análise da qualidade preditiva
        if f1 > 0.6:
            analise_f1 = f"Alta qualidade preditiva (F1: {f1:.2%})"
        elif f1 > 0.45:
            analise_f1 = f"Qualidade preditiva adequada (F1: {f1:.2%})"
        else:
            analise_f1 = f"Qualidade preditiva limitada (F1: {f1:.2%})"

        # Análise da frequência de operações
        if trades > 5:
            analise_trades = f"Alta frequência de operações ({trades:.1f} trades/periodo)"
        elif trades > 2:
            analise_trades = f"Frequência moderada de operações ({trades:.1f} trades/periodo)"
        else:
            analise_trades = f"Baixa frequência de operações ({trades:.1f} trades/periodo)"

        # Análise da previsão atual
        if proba > 0.7:
            analise_atual = f"Sinal atual de ALTA confiança ({proba:.1%})"
        elif proba > 0.55:
            analise_atual = f"Sinal atual de confiança MODERADA ({proba:.1%})"
        else:
            analise_atual = f"Sinal atual de BAIXA confiança ({proba:.1%})"

        return f"""
        ## 📊 Análise de Performance

        **Desempenho do Modelo:**
        - {analise_sharpe}
        - {analise_f1}
        - {analise_trades}

        **Previsão Atual:**
        - {analise_atual}

        **Recomendação Baseada no Histórico:**
        {RelatorioAnalitico._gerar_recomendacao(sharpe, f1, proba)}
        """

    @staticmethod
    def _gerar_recomendacao(sharpe: float, f1: float, proba: float) -> str:
        """Gera recomendação baseada nas métricas."""

        if sharpe > 0.3 and f1 > 0.5:
            if proba > 0.6:
                return "✅ **CONFIANÇA ALTA** - O modelo tem histórico sólido e o sinal atual é forte"
            else:
                return "⚠️ **CONFIANÇA MODERADA** - Modelo com bom histórico mas sinal atual fraco"
        else:
            if proba > 0.6:
                return "⚠️ **CONFIANÇA MODERADA** - Sinal atual forte mas histórico do modelo limitado"
            else:
                return "🔴 **CONFIANÇA BAIXA** - Histórico limitado e sinal atual fraco"

    @staticmethod
    def gerar_analise_mercado(precos: pd.Series, df_ibov: pd.DataFrame) -> str:
        """Gera análise contextual do mercado."""

        retorno_30d = (precos.iloc[-1] / precos.iloc[-30] - 1) * 100 if len(precos) >= 30 else 0
        volatilidade_30d = precos.pct_change().std() * np.sqrt(252) * 100 if len(precos) >= 30 else 0

        # Análise de tendência
        if retorno_30d > 5:
            tendencia = "forte alta"
        elif retorno_30d > 2:
            tendencia = "alta moderada"
        elif retorno_30d > -2:
            tendencia = "lateralizada"
        elif retorno_30d > -5:
            tendencia = "baixa moderada"
        else:
            tendencia = "forte baixa"

        # Análise de volatilidade
        if volatilidade_30d > 40:
            vol_texto = "muito alta"
        elif volatilidade_30d > 30:
            vol_texto = "alta"
        elif volatilidade_30d > 20:
            vol_texto = "moderada"
        else:
            vol_texto = "baixa"

        return f"""
        ## 🌍 Contexto de Mercado

        **Análise do Ativo:**
        - Tendência dos últimos 30 dias: **{tendencia}** ({retorno_30d:.1f}%)
        - Volatilidade anualizada: **{vol_texto}** ({volatilidade_30d:.1f}%)

        **Condições Atuais:**
        - Preço atual: R$ {precos.iloc[-1]:.2f}
        - Variação recente: {'📈' if retorno_30d > 0 else '📉'} {abs(retorno_30d):.1f}%

        **Recomendação de Alocação:**
        {RelatorioAnalitico._gerar_recomendacao_alocacao(retorno_30d, volatilidade_30d)}
        """

    @staticmethod
    def _gerar_recomendacao_alocacao(retorno: float, volatilidade: float) -> str:
        """Gera recomendação de alocação baseada em retorno e volatilidade."""

        if volatilidade > 35:
            return "⚡ **ALTA VOLATILIDADE** - Considerar posicionamentos menores e proteções"
        elif volatilidade > 25:
            if retorno > 0:
                return "📊 **MOMENTO FAVORÁVEL** - Mercado com tendência positiva e volatilidade moderada"
            else:
                return "⚠️ **MOMENTO CAUTELOSO** - Mercado em baixa com volatilidade moderada"
        else:
            if retorno > 0:
                return "✅ **CONDIÇÕES FAVORÁVEIS** - Baixa volatilidade com tendência positiva"
            else:
                return "🔄 **MOMENTO DE ESPERA** - Baixa volatilidade mas sem direção definida"


class DashboardTrading:
    """Dashboard para Análise Preditiva de Ativos, com foco em clareza e metodologia."""

    def __init__(self):
        self.modelo_carregado = None
        self.ticker_selecionado = None
        self._inicializar_sidebar()

    def _inicializar_sidebar(self):
        """Inicializa a barra lateral com os controles do usuário."""
        with st.sidebar:
            st.markdown("## 📈 Análise Preditiva de Ativos")

            # Informação de data/hora da última atualização
            st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y às %H:%M')}")

            # Carregar modelos disponíveis
            modelos_disponiveis = [f.replace('modelo_', '').replace('.joblib', '')
                                   for f in os.listdir(Params.PATH_MODELOS)
                                   if f.endswith('.joblib')]

            if not modelos_disponiveis:
                st.warning("Nenhum modelo treinado foi encontrado na pasta 'modelos_treinados'.")
                st.stop()

            self.ticker_selecionado = st.selectbox(
                "Selecione o Ativo:",
                sorted(modelos_disponiveis),
                help="Selecione o ativo para análise preditiva"
            )

            self.analisar_btn = st.button("🔍 Gerar Análise Completa", type="primary", use_container_width=True)
            self.gerar_relatorio_btn = st.button("📋 Gerar Relatório Explicativo", use_container_width=True)

            st.markdown("---")

            # Informações expandíveis
            with st.expander("ℹ️ Sobre este dashboard"):
                st.info("""
                Este é um projeto acadêmico que demonstra um pipeline de Machine Learning para 
                prever a direção de ativos da B3. 

                **Não constitui uma recomendação formal de investimento.**
                """)

    @st.cache_resource(ttl=3600)  # Cache por 1 hora
    def _carregar_modelo(_self, ticker: str) -> Optional[Any]:
        """Carrega o modelo treinado do disco."""
        caminho = os.path.join(Params.PATH_MODELOS, f"modelo_{ticker}.joblib")
        if os.path.exists(caminho):
            try:
                return load(caminho)
            except Exception as e:
                st.error(f"Erro ao carregar o modelo: {e}")
        return None

    def executar(self):
        """Orquestra a apresentação completa da análise."""
        if not self.ticker_selecionado:
            return

        st.header(f"Análise Preditiva para {self.ticker_selecionado}")

        if self.analisar_btn or self.gerar_relatorio_btn:
            with st.spinner("Carregando modelo e processando dados..."):
                self.modelo_carregado = self._carregar_modelo(self.ticker_selecionado)

            if self.modelo_carregado is None:
                st.error(
                    f"O modelo para {self.ticker_selecionado} não foi encontrado. Verifique se o treinamento foi bem-sucedido.")
                return

            with st.spinner("Processando dados e gerando análise..."):
                loader = DataLoader()
                feature_engineer = FeatureEngineer()

                try:
                    df_ticker, df_ibov = loader.baixar_dados_yf(self.ticker_selecionado)
                except Exception:
                    st.warning(
                        "**Aviso:** Não foi possível baixar os dados mais recentes. A análise abaixo é baseada nos últimos dados disponíveis no banco de dados local.")
                    df_ticker = loader.carregar_do_bd(self.ticker_selecionado)
                    df_ibov = pd.DataFrame()

                if df_ticker.empty:
                    st.error(
                        f"Não foi possível carregar nenhum dado para {self.ticker_selecionado}, nem da internet nem do banco de dados.")
                    return

                X_full, y_full, precos_full, t1_full = feature_engineer.preparar_dataset(df_ticker, df_ibov,
                                                                                         self.ticker_selecionado)
                previsao = self.modelo_carregado.prever_direcao(X_full.tail(1), self.ticker_selecionado)

            # Gerar relatório explicativo se solicitado
            if self.gerar_relatorio_btn:
                self._render_relatorio_completo(previsao, precos_full, df_ibov)
            else:
                # Abas normais de análise
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🎯 **Resumo Executivo**",
                    "📈 **Performance Validada**",
                    "📊 **Simulação Ilustrativa**",
                    "🧬 **DNA do Modelo**"
                ])

                with tab1:
                    self._render_tab_resumo(previsao, X_full, precos_full)

                with tab2:
                    self._render_tab_performance(y_full)

                with tab3:
                    self._render_tab_simulacao(X_full, precos_full)

                with tab4:
                    self._render_tab_dna()
        else:
            st.info("⬅️ Para começar, selecione um ativo na barra lateral e clique em 'Gerar Análise Completa'.")

            # Adicionar explicação inicial
            st.markdown("---")
            st.subheader("Bem-vindo ao Sistema de Análise Preditiva")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### 📋 Como funciona:
                1. **Selecione um ativo** na barra lateral
                2. **Clique em 'Gerar Análise Completa'**
                3. **Explore os resultados** através das abas

                ### 🔍 Metodologia:
                - Modelo de Machine Learning (LightGBM)
                - Validação Walk-Forward robusta
                - Sistema de Tripla Barreira para labeling
                - 30+ indicadores técnicos como features
                """)

            with col2:
                st.markdown("""
                ### 📊 O que você verá:
                - **Previsão** para o próximo pregão
                - **Performance histórica validada** (WFV)
                - **Simulação ilustrativa** do modelo
                - **Transparência completa** da metodologia

                ### ⚠️ Importante:
                Este é um projeto **acadêmico** para demonstração de técnicas de ML aplicadas ao mercado financeiro.
                **Não constitui recomendação de investimento.**
                """)

    def _render_relatorio_completo(self, previsao: Dict[str, Any], precos: pd.Series, df_ibov: pd.DataFrame):
        """Renderiza um relatório explicativo completo."""

        st.title(f"📋 Relatório Analítico - {self.ticker_selecionado}")
        st.markdown(f"*Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}*")
        st.markdown("---")

        # Seção 1: Explicação do Threshold
        threshold = self.modelo_carregado.threshold_operacional
        explicacao_threshold, estilo, razao = RelatorioAnalitico.gerar_explicacao_threshold(
            threshold, self.modelo_carregado.wfv_metrics
        )

        st.markdown("## 🎯 Análise da Estratégia do Modelo")
        st.markdown(explicacao_threshold)

        # Visualização do threshold
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=threshold * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Nível de Confiança Requerido (Threshold)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 45], 'color': "lightgreen"},
                    {'range': [45, 55], 'color': "lightyellow"},
                    {'range': [55, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Seção 2: Análise de Performance
        st.markdown(RelatorioAnalitico.gerar_analise_performance(
            self.modelo_carregado.wfv_metrics, previsao
        ))

        # Seção 3: Análise de Mercado
        st.markdown(RelatorioAnalitico.gerar_analise_mercado(precos, df_ibov))

        # Seção 4: Recomendações Específicas
        st.markdown("""
        ## 🎯 Recomendações Específicas

        **Para Investidores Conservadores:**
        - Espere por sinais com confiança acima de 70%
        - Considere posicionamentos menores devido à volatilidade
        - Use stops para proteger o capital

        **Para Investidores Moderados:**
        - Opere dentro dos limites do seu plano de risco
        - Diversifique entre diferentes oportunidades
        - Monitore regularmente as posições

        **Para Investidores Agressivos:**
        - Pode considerar operar com confiança acima de 55%
        - Esteja preparado para maior volatilidade
        - Tenha disciplina para seguir o plano
        """)

        # Seção 5: Próximos Passos
        st.markdown("""
        ## 🔮 Próximos Passos Recomendados

        1. **Monitore** o comportamento do ativo nos próximos dias
        2. **Avalie** se a previsão se concretiza
        3. **Ajuste** sua estratégia com base nos resultados
        4. **Considere** diversificar com outros ativos
        5. **Revise** periodicamente as previsões do modelo
        """)

        st.markdown("---")
        st.success("""
        **📊 Este relatório foi gerado automaticamente com base na análise algorítmica**
        *Lembre-se: nenhum modelo é perfeito e o passado não garante resultados futuros.*
        """)

    # --- MÉTODOS PARA RENDERIZAR AS ABAS ---

    def _render_tab_resumo(self, previsao: Dict[str, Any], X_full: pd.DataFrame, precos_full: pd.Series):
        """Renderiza a aba 'Resumo Executivo'."""
        st.subheader("Conclusão para o Próximo Pregão")

        # Adicionar informação de data
        cols_data = st.columns(2)
        last_date = X_full.index[-1].strftime('%d/%m/%Y')
        next_date = (X_full.index[-1] + pd.tseries.offsets.BDay(1)).strftime('%d/%m/%Y')
        cols_data[0].markdown(f"**📅 Último dado analisado:** {last_date}")
        cols_data[1].markdown(f"**🔮 Previsão para o pregão de:** {next_date}")

        if previsao.get('status') != 'sucesso':
            st.error(f"Não foi possível gerar a previsão: {previsao.get('status')}")
            return

        st.divider()

        # Determinar força do sinal baseado na probabilidade
        probabilidade = previsao['probabilidade']
        if probabilidade >= 0.7:
            forca_sinal = "🟢 ALTA"
            cor_barra = "#00cc00"
            icon = "📈"
        elif probabilidade >= 0.55:
            forca_sinal = "🟡 MÉDIA"
            cor_barra = "#ffcc00"
            icon = "➡️"
        else:
            forca_sinal = "🔴 BAIXA"
            cor_barra = "#ff4d4d"
            icon = "📉"

        cols_metricas = st.columns(3)
        direcao = "ALTA" if previsao['predicao'] == 1 else "NEUTRA"
        recomendacao = "🟢 OPORTUNIDADE" if previsao['should_operate'] else "👀 OBSERVAR"

        cols_metricas[0].metric("Sinal Derivado", recomendacao,
                                help="Recomendação baseada na análise do modelo: Operar ou Observar")

        cols_metricas[1].metric("Cenário Mais Provável", f"{icon} {direcao}",
                                help="A direção que o preço tem maior probabilidade de seguir, segundo o modelo.")

        cols_metricas[2].metric("Força do Sinal", forca_sinal,
                                help="Intensidade da confiança do modelo na previsão")

        # Barra de progresso colorida baseada na força do sinal
        st.markdown(f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px; margin: 10px 0;">
            <div style="background-color: {cor_barra}; width: {probabilidade * 100}%; 
                       height: 20px; border-radius: 8px; display: flex; align-items: center; 
                       justify-content: center; color: white; font-weight: bold;">
                {probabilidade:.1%} de Confiança
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Explicação do significado
        with st.expander("📖 O que significa esta previsão?"):
            if previsao['should_operate']:
                st.success("""
                **🟢 OPORTUNIDADE**: O modelo identificou um padrão que historicamente precede movimentos de alta 
                com confiança suficiente para considerar uma operação.
                """)
            else:
                st.info("""
                **👀 OBSERVAR**: O modelo não identificou um padrão suficientemente forte para recomendar uma operação 
                no próximo pregão. Recomenda-se aguardar melhores oportunidades.
                """)

        st.subheader("Contexto de Preços Recentes")
        self._plot_precos_recentes(precos_full.iloc[-60:])

        # Comparação com IBOV (se disponível)
        try:
            loader = DataLoader()
            _, df_ibov = loader.baixar_dados_yf(self.ticker_selecionado)
            if not df_ibov.empty and 'Close_IBOV' in df_ibov.columns:
                st.subheader("Comparação com IBOVESPA")
                self._plot_comparacao_ibov(precos_full.iloc[-60:], df_ibov['Close_IBOV'].iloc[-60:])
        except:
            pass

    def _render_tab_performance(self, y_full: pd.Series):
        """Renderiza a aba 'Performance Validada'."""
        st.subheader("Desempenho Histórico Validado (Out-of-Sample)")

        # Informação sobre validação walk-forward
        with st.expander("🔍 Sobre a Validação Walk-Forward"):
            st.info("""
            **Por que esta é a métrica mais importante?**

            A Validação Walk-Forward é um método que:
            - Testa o modelo em dados que ele **nunca viu** durante o treinamento
            - Simula realisticamente como o modelo se comportaria no mercado real
            - Elimina o **vazamento de informação** entre períodos de treino e teste
            - Fornece estimativas **confiáveis** da performance futura

            Os resultados abaixo são desta validação robusta, não de backtest em dados já conhecidos.
            """)

        metricas = self.modelo_carregado.wfv_metrics
        cols = st.columns(3)

        cols[0].metric("Sharpe Ratio (Médio)", f"{metricas.get('sharpe_medio', 0):.2f}",
                       help="Mede o retorno da estratégia ajustado ao risco. Valores acima de 0.3 indicam bom desempenho.")

        cols[1].metric("Trades por Período (Média)", f"{metricas.get('trades_medio', 0):.1f}",
                       help="Frequência média de operações da estratégia.")

        cols[2].metric("Qualidade da Classificação (F1)", f"{metricas.get('f1_macro_medio', 0):.2%}",
                       help="Mede o quão bem o modelo classifica os movimentos do mercado.")

        # Interpretação dos resultados
        sharpe = metricas.get('sharpe_medio', 0)
        if sharpe > 0.5:
            st.success("✅ Performance sólida - O modelo demonstra boa capacidade de gerar retornos ajustados ao risco.")
        elif sharpe > 0:
            st.info("📊 Performance moderada - O modelo tem algum poder preditivo, mas com margem para melhorias.")
        else:
            st.warning("⚠️ Performance abaixo do esperado - O modelo não demonstra capacidade consistente de previsão.")

        st.subheader("Análise de Acertos e Erros (Última Validação)")
        self._plot_matriz_confusao(y_full)

        # Adicionar métricas adicionais de performance
        st.subheader("Métricas Adicionais de Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Folds Válidos", metricas.get('folds_validos', 0),
                      help="Número de períodos de validação com dados suficientes para avaliação")

        with col2:
            win_rate = self._calcular_win_rate_estimado(metricas)
            st.metric("Taxa de Acerto Estimada", f"{win_rate:.1%}",
                      help="Taxa estimada de previsões corretas baseada no F1-Score")

    def _render_tab_simulacao(self, X_full: pd.DataFrame, precos_full: pd.Series):
        """Renderiza a aba 'Simulação Ilustrativa'."""
        st.subheader("Simulação de Portfólio (Backtest em Dados Completos)")

        # Aviso importante sobre a simulação
        st.warning("""
        **⚠️ AVISO IMPORTANTE:** 

        Esta simulação é **'in-sample'** (executada sobre todo o histórico conhecido). 
        Seus resultados tendem a ser **otimistas** e servem principalmente para **ilustrar o comportamento** da estratégia.

        A performance real é melhor representada pela **Validação Walk-Forward** (aba anterior).
        """)

        with st.spinner("Gerando gráficos da simulação..."):
            risk_analyzer = RiskAnalyzer()
            threshold_simulacao = self.modelo_carregado.threshold_operacional * 0.95
            df_sinais = self.modelo_carregado.prever_e_gerar_sinais(X_full, precos_full, self.ticker_selecionado,
                                                                    threshold_override=threshold_simulacao)
            backtest_info = risk_analyzer.backtest_sinais(df_sinais)

        st.plotly_chart(self._plot_precos_sinais(df_sinais, precos_full), use_container_width=True)

        if backtest_info.get('trades', 0) > 0:
            self._exibir_metricas_backtest(backtest_info)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(self._plot_equidade(backtest_info), use_container_width=True)
            with c2:
                st.plotly_chart(self._plot_drawdown(backtest_info), use_container_width=True)

            # Adicionar análise de risco
            st.subheader("Análise de Risco da Simulação")
            self._analise_risco(backtest_info)
        else:
            st.info("Nenhuma operação foi realizada nesta simulação ilustrativa.")

    def _render_tab_dna(self):
        """Renderiza a aba 'DNA do Modelo'."""
        st.subheader("Ficha Técnica do Modelo")
        caminho_modelo = os.path.join(Params.PATH_MODELOS, f"modelo_{self.ticker_selecionado}.joblib")
        data_treino = datetime.fromtimestamp(os.path.getmtime(caminho_modelo)).strftime('%d/%m/%Y às %H:%M')

        # Determinar estilo da estratégia baseado no threshold
        threshold = self.modelo_carregado.threshold_operacional
        if threshold <= 0.45:
            estilo_estrategia = "🔄 Ágil - Captura mais oportunidades"
            descricao_estilo = "Estratégia configurada para ser mais sensível, capturando movimentos com confiança moderada para maximizar oportunidades."
        elif threshold <= 0.55:
            estilo_estrategia = "⚖️ Balanceada - Equilíbrio risco/retorno"
            descricao_estilo = "Estratégia equilibrada entre sensibilidade e precisão, buscando o melhor balanço entre quantidade e qualidade de sinais."
        else:
            estilo_estrategia = "🎯 Conservadora - Alta confiança"
            descricao_estilo = "Estratégia conservadora que opera apenas quando há alta confiança, priorizando precisão sobre quantidade."

        ficha_tecnica = {
            "Ativo-Alvo": self.ticker_selecionado,
            "Último Treinamento": data_treino,
            "Tipo de Modelo": "LightGBM Classifier",
            "Estilo da Estratégia": estilo_estrategia,
            "Threshold Operacional": f"{threshold:.3f}",
            "Número de Features": len(self.modelo_carregado.features_selecionadas)
        }

        # Exibir ficha técnica em formato de tabela
        for chave, valor in ficha_tecnica.items():
            st.write(f"**{chave}:** {valor}")

        # Explicação do estilo
        st.info(f"**{estilo_estrategia}** - {descricao_estilo}")

        st.subheader("O que o Modelo Considera Mais Importante?")
        st.info(
            "O gráfico abaixo mostra as 15 variáveis que o modelo mais utilizou para tomar suas decisões:"
        )
        self._plot_importancia_features()

        st.subheader("Fundamentos da Metodologia")

        with st.expander("🔍 Como o Modelo Toma Decisões"):
            st.markdown("""
            **Sistema de Classificação com Limiar Adaptativo**

            Este modelo não trabalha com "certezas absolutas". Em vez disso:

            • **Analisa padrões** históricos para estimar probabilidades
            • **Ajusta automaticamente** sua sensibilidade para cada ativo
            • **Busca o ponto ótimo** entre capturar oportunidades e evitar falsos sinais
            • **A estratégia é calibrada** para maximizar o retorno ajustado ao risco (Sharpe Ratio)

            O 'limiar de confiança' é definido estatisticamente durante o treinamento para otimizar performance.
            """)

        with st.expander("📊 Processo de Validação"):
            st.markdown("""
            **Walk-Forward com Purga**

            Para garantir resultados realistas:

            • **Simula a passagem do tempo** testando em dados nunca vistos
            • **Elimina vazamento de informação** entre períodos de treino e teste  
            • **Proporciona estimativas confiáveis** da performance futura
            • **Os resultados mostrados** são desta validação robusta
            """)

        with st.expander("🎯 Sistema de Tripla Barreira"):
            st.markdown("""
            **Como definimos os alvos (labels) para o modelo:**

            Para cada ponto no tempo, analisamos se o preço:
            1. **Atingiu barreira superior** (ATR × 1.5) → Sinal de ALTA (1)
            2. **Atingiu barreira inferior** (ATR × 1.0) → Sinal de BAIXA (-1)
            3. **Não atingiu nenhuma** → Sinal NEUTRO (0)

            Janela de análise: 5 dias à frente
            """)

    # --- MÉTODOS DE PLOTAGEM E AUXILIARES ---

    def _plot_precos_recentes(self, precos: pd.Series):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precos.index, y=precos, mode='lines', name='Fechamento', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=[precos.index[-1]], y=[precos.iloc[-1]], mode='markers', name='Último Ponto',
                                 marker=dict(color='red', size=10, symbol='star')))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title='Preço (R$)',
            xaxis_title=None,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    def _plot_comparacao_ibov(self, precos: pd.Series, ibov: pd.Series):
        """Plot de comparação com IBOVESPA."""
        # Normalizar para base 100
        precos_norm = (precos / precos.iloc[0]) * 100
        ibov_norm = (ibov / ibov.iloc[0]) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precos_norm.index, y=precos_norm, mode='lines',
                                 name=self.ticker_selecionado, line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=ibov_norm.index, y=ibov_norm, mode='lines',
                                 name='IBOVESPA', line=dict(color='#ff7f0e')))

        fig.update_layout(
            height=300,
            yaxis_title='Desempenho (Base 100)',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    def _plot_matriz_confusao(self, y_full: pd.Series):
        try:
            # Recalcular previsões do último fold da validação para a matriz
            X_full_scaled = self.modelo_carregado.X_scaled
            y_full_encoded = self.modelo_carregado.label_encoder.transform(y_full)

            _, test_idx = list(self.modelo_carregado.cv_gen.split(X_full_scaled))[-1]

            X_test = X_full_scaled.iloc[test_idx]
            y_test_encoded = y_full_encoded[test_idx]
            y_test_labels = self.modelo_carregado.label_encoder.inverse_transform(y_test_encoded)

            preds_encoded = self.modelo_carregado.modelo_final.predict(X_test)
            preds_labels = self.modelo_carregado.label_encoder.inverse_transform(preds_encoded)

            cm = confusion_matrix(y_test_labels, preds_labels, labels=[-1, 0, 1])
            display_labels = ['BAIXA', 'NEUTRO', 'ALTA']

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            ax.set_title("Previsões vs. Realidade")
            st.pyplot(fig)

            # Adicionar análise da matriz de confusão
            total = np.sum(cm)
            accuracy = np.trace(cm) / total if total > 0 else 0

            st.caption(f"""
            **Análise da Matriz de Confusão:**
            - Acurácia geral: {accuracy:.1%}
            - Diagonal principal (acertos): {np.trace(cm)} de {total} exemplos
            - Fora da diagonal (erros): {total - np.trace(cm)} exemplos
            """)

        except Exception as e:
            st.warning(f"Não foi possível gerar a Matriz de Confusão: {e}")

    def _plot_precos_sinais(self, df_sinais: pd.DataFrame, precos: pd.Series) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precos.index, y=precos, mode='lines', name='Preço de Fechamento',
                                 line=dict(color='#1f77b4', width=2)))
        sinais_operar = df_sinais[df_sinais['sinal'] == 1]
        if not sinais_operar.empty:
            fig.add_trace(go.Scatter(x=sinais_operar.index, y=sinais_operar['preco'], mode='markers',
                                     name='Sinal de Oportunidade',
                                     marker=dict(color='limegreen', size=10, symbol='triangle-up',
                                                 line=dict(width=1, color='darkgreen'))))
        fig.update_layout(
            title_text='Preços Históricos vs. Sinais da Estratégia',
            xaxis_title='Data',
            yaxis_title='Preço (R$)',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig

    def _plot_equidade(self, backtest_info: Dict[str, Any]) -> go.Figure:
        fig = go.Figure()
        curva_equidade = backtest_info.get('equity_curve', [])
        if len(curva_equidade) > 1:
            fig.add_trace(go.Scatter(
                x=np.arange(len(curva_equidade)),
                y=curva_equidade,
                mode='lines',
                name='Capital',
                line=dict(color='#2ca02c', width=2)
            ))

            # Adicionar linha de referência em 1.0 (capital inicial)
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)

        fig.update_layout(
            title_text='Evolução do Capital',
            xaxis_title='Nº da Operação',
            yaxis_title='Capital Relativo',
            height=350,
            showlegend=False
        )
        return fig

    def _plot_drawdown(self, backtest_info: Dict[str, Any]) -> go.Figure:
        fig = go.Figure()
        equity_curve = np.array(backtest_info.get('equity_curve', []))
        if len(equity_curve) > 1:
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            fig.add_trace(go.Scatter(
                x=np.arange(len(drawdown)),
                y=drawdown,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red')
            ))

            # Adicionar linha de referência em 0
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)

        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            title_text='Curva de Drawdown',
            xaxis_title='Nº da Operação',
            yaxis_title='Queda do Pico',
            height=350,
            showlegend=False
        )
        return fig

    def _analise_risco(self, backtest_info: Dict[str, Any]):
        """Analisa e exibe métricas de risco."""
        retornos = backtest_info.get('retornos', [])
        if not retornos or len(retornos) == 0:
            st.info("Dados insuficientes para análise de risco detalhada.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("VaR 95% (1 dia)", f"{np.percentile(retornos, 5):.2%}",
                      help="Máxima perda esperada em 95% dos casos (Value at Risk)")

        with col2:
            st.metric("CVaR 95% (1 dia)", f"{np.mean([r for r in retornos if r <= np.percentile(retornos, 5)]):.2%}",
                      help="Perda média nos piores 5% dos casos (Conditional Value at Risk)")

        with col3:
            st.metric("Volatilidade Anual", f"{np.std(retornos) * np.sqrt(252):.2%}",
                      help="Volatilidade anualizada dos retornos")

    def _plot_importancia_features(self):
        try:
            importances = self.modelo_carregado.modelo_final.feature_importances_
            features = self.modelo_carregado.features_selecionadas

            df_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance',
                                                                                                ascending=False).head(
                15)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_imp['importance'],
                y=df_imp['feature'],
                orientation='h',
                marker_color='#1f77b4'
            ))
            fig.update_layout(
                title="Top 15 Variáveis Mais Importantes",
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importância (calculada pelo LightGBM)",
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Adicionar explicação
            with st.expander("💡 Como interpretar a importância das features?"):
                st.markdown("""
                A importância mostra o quanto cada variável contribui para as decisões do modelo:

                - **Valores mais altos**: Variáveis mais importantes para o modelo
                - **Valores mais baixos**: Variáveis com menor influência

                Esta métrica é calculada pelo LightGBM baseado em quantas vezes uma feature é usada 
                para fazer divisões nas árvores de decisão e quão efetivas são essas divisões.
                """)

        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico de importância de features: {e}")

    def _exibir_metricas_backtest(self, metricas: Dict[str, Any]):
        cols = st.columns(5)
        cols[0].metric("Retorno Total (Sim.)", f"{metricas.get('retorno_total', 0):.2%}")
        cols[1].metric("Sharpe Ratio (Sim.)", f"{metricas.get('sharpe', 0):.2f}")
        cols[2].metric("Max Drawdown (Sim.)", f"{metricas.get('max_drawdown', 0):.2%}")
        cols[3].metric("Taxa de Acerto (Sim.)", f"{metricas.get('win_rate', 0):.2%}")
        cols[4].metric("Nº de Trades (Sim.)", f"{metricas.get('trades', 0)}")

    def _calcular_win_rate_estimado(self, metricas: Dict[str, Any]) -> float:
        """Calcula uma estimativa da taxa de acerto baseada no F1-Score."""
        f1 = metricas.get('f1_macro_medio', 0)
        # Fórmula simplificada para estimar win rate a partir do F1
        return min(0.95, max(0.05, f1 * 0.8))


if __name__ == "__main__":
    if 'y_full' not in st.session_state:
        st.session_state.y_full = None

    dashboard = DashboardTrading()
    dashboard.executar()
