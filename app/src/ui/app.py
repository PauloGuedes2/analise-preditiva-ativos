import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load

from src.config.params import Params
from src.data.data_loader import DataLoader
from src.models.feature_engineer import FeatureEngineer
from src.utils.risk_analyzer import RiskAnalyzer

st.set_page_config(layout="wide", page_title="Análise Preditiva de Ativos", page_icon="📈")


class DashboardTrading:
    """Dashboard para Análise Preditiva de Ativos, com foco em clareza e metodologia."""

    def __init__(self):
        self.modelo_carregado = None
        self._inicializar_sidebar()

    def _inicializar_sidebar(self):
        """Inicializa a barra lateral com os controles do usuário."""
        with st.sidebar:
            st.markdown("---")

            self.ticker_selecionado = st.selectbox("Selecione o Ativo:", Params.TICKERS)

            st.info(
                "Este dashboard apresenta os resultados de um modelo de Inteligência Artificial treinado para prever a direção de ativos da B3.")

            self.analisar_btn = st.button("Gerar Análise Completa", type="primary", use_container_width=True)

            st.markdown("---")
            st.write("Um projeto de análise quantitativa.")

    @st.cache_resource
    def _carregar_modelo(_self, ticker: str) -> Optional[Any]:
        """Carrega o modelo treinado do disco."""
        caminho = os.path.join(Params.PATH_MODELOS, f"modelo_{ticker}.joblib")
        if os.path.exists(caminho):
            try:
                return load(caminho)
            except Exception as e:
                st.error(f"Erro ao carregar o modelo: {e}")
        return None

    def _exibir_conclusao_analise(self, previsao: Dict[str, Any]):
        """Seção 1: A Conclusão da Análise para o próximo dia."""
        st.header(f"Conclusão da Análise para {self.ticker_selecionado}")

        if previsao.get('status') != 'sucesso':
            st.error(f"Não foi possível gerar a previsão: {previsao.get('status')}")
            return

        st.markdown("Esta seção resume a previsão do modelo para o próximo pregão, com base nos dados mais recentes.")

        cols = st.columns(3)
        direcao = "ALTA" if previsao['predicao'] == 1 else "NEUTRA"
        recomendacao = "OPORTUNIDADE" if previsao['should_operate'] else "OBSERVAR"

        cols[0].metric("Sinal Derivado", f"**{recomendacao}**",
                       help="'OPORTUNIDADE' indica que a confiança do modelo na previsão de ALTA superou seu limiar de decisão, que foi calibrado estatisticamente.")
        cols[1].metric("Cenário Mais Provável", direcao,
                       help="A direção (Alta ou Neutra/Baixa) que o preço da ação tem maior probabilidade de seguir, segundo o modelo.")
        cols[2].metric("Probabilidade de Alta", f"{previsao['probabilidade']:.1%}",
                       help=f"A probabilidade estimada pelo modelo para o cenário de ALTA. O sinal 'OPORTUNIDADE' é gerado se este valor ultrapassar o limiar calibrado de {previsao['threshold_operacional']:.1%}.")

    def _exibir_performance_validada(self, metricas: Dict[str, Any]):
        """Seção 2: A Performance Validada do Modelo."""
        st.header("Desempenho Histórico Validado (Out-of-Sample)")
        st.info(
            "Estes são os resultados mais importantes. Eles foram calculados durante a **Validação Walk-Forward**, um método rigoroso que testa o modelo em dados que ele nunca viu, simulando sua performance em condições reais.")

        cols = st.columns(3)
        cols[0].metric("Sharpe Ratio (Médio)", f"{metricas.get('sharpe_medio', 0):.2f}",
                       help="Mede o retorno da estratégia ajustado ao seu risco. Valores acima de 0.3 indicam um desempenho consistentemente lucrativo nos testes. É a principal métrica de performance financeira.")
        cols[1].metric("Trades por Período (Média)", f"{metricas.get('trades_medio', 0):.1f}",
                       help="A frequência média de operações da estratégia durante a validação. Indica se o modelo é mais seletivo ou mais ativo.")
        cols[2].metric("Qualidade da Classificação (F1)", f"{metricas.get('f1_macro_medio', 0):.2%}",
                       help="Mede o quão bem o modelo consegue classificar corretamente os movimentos do mercado (alta, baixa, neutro).")

    def _exibir_simulacao_completa(self, X_full, precos_full):
        """Seção 3: A Simulação Completa (In-Sample)."""
        st.header("Simulação de Portfólio (Backtest em Dados Completos)")
        st.warning(
            "**Aviso Metodológico:** A simulação abaixo é **'in-sample'** (executada sobre todo o histórico de dados). Seus resultados tendem a ser otimistas e servem principalmente para **ilustrar o comportamento** da estratégia. A performance real é melhor representada pelos resultados da **Validação Walk-Forward** acima.")

        with st.spinner("Gerando gráficos da simulação..."):
            risk_analyzer = RiskAnalyzer()
            df_sinais = self.modelo_carregado.prever_e_gerar_sinais(X_full, precos_full, self.ticker_selecionado)
            backtest_info = risk_analyzer.backtest_sinais(df_sinais)

        fig_precos = self._criar_grafico_precos_sinais(df_sinais, precos_full)
        st.plotly_chart(fig_precos, use_container_width=True)

        if backtest_info.get('trades', 0) > 0:
            fig_equity = self._criar_grafico_equidade(backtest_info)
            st.plotly_chart(fig_equity, use_container_width=True)
            self._exibir_metricas_backtest(backtest_info)
        else:
            st.info("Nenhuma operação foi realizada nesta simulação.")

    def _exibir_fundamentos_modelo(self):
        """Seção 4: A parte educativa e de explicabilidade."""
        st.header("Fundamentos do Modelo Preditivo")
        st.write("Esta seção oferece um resumo técnico e conceitual de como o modelo foi construído e como ele opera.")

        with st.expander("Metodologia: Classificação vs. Regressão"):
            st.markdown("""
            No campo de Machine Learning, existem duas abordagens principais para previsões numéricas:
            - **Regressão:** Tenta prever um **valor contínuo e exato**. (Ex: "O preço da ação será R$ 34,52").
            - **Classificação:** Tenta prever uma **categoria ou classe**. (Ex: "O movimento da ação será de 'ALTA', 'BAIXA' ou 'NEUTRO'?").

            **Este projeto utiliza um modelo de CLASSIFICAÇÃO.** O objetivo não é acertar o preço exato, mas sim prever a **direção** do movimento, uma tarefa mais robusta para a tomada de decisão.
            """)

        with st.expander("Inteligência do Modelo: As Features Utilizadas"):
            st.markdown("""
            Para tomar suas decisões, o modelo analisa um conjunto de 'features' (variáveis) extraídas dos dados históricos. As mais importantes para este ativo, selecionadas dinamicamente durante o último treinamento, foram:
            """)
            df_features = pd.DataFrame(self.modelo_carregado.features_selecionadas,
                                       columns=['Variáveis Mais Relevantes (Features)'])
            st.dataframe(df_features, use_container_width=True, hide_index=True)

        with st.expander("Processo de Aprendizado e Validação"):
            st.markdown("""
            O modelo foi treinado e validado com métodos rigorosos para garantir sua relevância e evitar otimismo excessivo:
            1.  **Treinamento:** O modelo analisa dados históricos para aprender a correlação entre as *features* e os movimentos futuros de preço.
            2.  **Validação (Walk-Forward com Purga):** Para testar o aprendizado, aplicamos um método que simula a passagem do tempo. O modelo é treinado em um período e testado em um período futuro, em um processo que se repete. Crucialmente, usamos uma "purga" para garantir que não haja sobreposição de informações entre treino e teste, um cuidado essencial em dados financeiros.

            As métricas na seção **'Desempenho Histórico Validado'** são o resultado direto deste processo de validação rigoroso.
            """)

    def executar(self):
        """Orquestra a apresentação completa da análise."""
        if self.analisar_btn:
            st.title(f"Análise Preditiva: {self.ticker_selecionado}")
            self.modelo_carregado = self._carregar_modelo(self.ticker_selecionado)

            if self.modelo_carregado is None:
                st.error(
                    f"O modelo para {self.ticker_selecionado} não foi encontrado. Ele pode não ter atingido os critérios mínimos de performance durante o treinamento para ser salvo.")
                return

            with st.spinner("Processando dados e gerando análise..."):
                loader = DataLoader()
                feature_engineer = FeatureEngineer()
                df_ticker, df_ibov = loader.baixar_dados_yf(self.ticker_selecionado)
                X_full, _, precos_full, _ = feature_engineer.preparar_dataset(df_ticker, df_ibov,
                                                                              self.ticker_selecionado)
                previsao = self.modelo_carregado.prever_direcao(X_full.tail(1), self.ticker_selecionado)

            # --- Estrutura Lógica do Dashboard ---
            self._exibir_conclusao_analise(previsao)
            st.divider()
            if hasattr(self.modelo_carregado, 'wfv_metrics') and self.modelo_carregado.wfv_metrics:
                self._exibir_performance_validada(self.modelo_carregado.wfv_metrics)
            st.divider()
            if not X_full.empty:
                self._exibir_simulacao_completa(X_full, precos_full)
            st.divider()
            self._exibir_fundamentos_modelo()
            st.divider()
            st.caption("Este é um projeto acadêmico e não constitui uma recomendação formal de investimento.")
        else:
            st.info("⬆️ Para começar, selecione um ativo na barra lateral e clique em 'Gerar Análise Completa'.")

    def _criar_grafico_precos_sinais(self, df_sinais: pd.DataFrame, precos: pd.Series) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precos.index, y=precos, mode='lines', name='Preço de Fechamento',
                                 line=dict(color='#1f77b4', width=2)))
        sinais_operar = df_sinais[df_sinais['sinal'] == 1]
        if not sinais_operar.empty:
            fig.add_trace(go.Scatter(x=sinais_operar.index, y=sinais_operar['preco'], mode='markers',
                                     name='Sinal de Oportunidade',
                                     marker=dict(color='limegreen', size=10, symbol='triangle-up',
                                                 line=dict(width=1, color='darkgreen'))))
        fig.update_layout(title_text='Preços Históricos vs. Sinais da Estratégia (Simulação In-Sample)',
                          xaxis_title='Data', yaxis_title='Preço (R$)', hovermode='x unified',
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    def _criar_grafico_equidade(self, backtest_info: Dict[str, Any]) -> go.Figure:
        fig = go.Figure()
        curva_equidade = backtest_info.get('equity_curve', [])
        if len(curva_equidade) > 1:
            fig.add_trace(
                go.Scatter(x=np.arange(len(curva_equidade)), y=curva_equidade, mode='lines', name='Curva de Capital',
                           line=dict(color='#2ca02c', width=3)))
        fig.update_layout(title_text='Evolução do Capital na Simulação', xaxis_title='Nº da Operação',
                          yaxis_title='Capital Relativo (Início = 1.0)', hovermode='x unified')
        return fig

    def _exibir_metricas_backtest(self, metricas: Dict[str, Any]):
        st.subheader("Métricas Detalhadas da Simulação")
        cols = st.columns(5)
        cols[0].metric("Retorno Total", f"{metricas.get('retorno_total', 0):.2%}",
                       help="O retorno acumulado da estratégia do início ao fim da simulação.")
        cols[1].metric("Sharpe Ratio (Otimista)", f"{metricas.get('sharpe', 0):.2f}",
                       help="Mede o retorno ajustado ao risco. (Lembre-se: este valor é otimista por ser in-sample).")
        cols[2].metric("Max Drawdown", f"{metricas.get('max_drawdown', 0):.2%}",
                       help="A maior queda percentual do pico ao fundo durante a simulação. É uma medida chave de risco.")
        cols[3].metric("Taxa de Acerto", f"{metricas.get('win_rate', 0):.2%}",
                       help="Das operações que o modelo fez nesta simulação, qual a porcentagem que deu lucro.")
        cols[4].metric("Nº de Trades", f"{metricas.get('trades', 0)}",
                       help="O número total de operações (compra e venda) realizadas na simulação.")


if __name__ == "__main__":
    dashboard = DashboardTrading()
    dashboard.executar()
