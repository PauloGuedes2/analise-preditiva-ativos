import os
from typing import Optional, Dict, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load

from src.data.data_loader import DataLoader
from src.models.feature_engineer import FeatureEngineer
from src.utils.risk_analyzer import RiskAnalyzer


class DashboardTrading:
    """Dashboard interativo para análise preditiva de ativos."""

    def __init__(self):
        self._configurar_pagina()
        self._inicializar_sidebar()

    @staticmethod
    def _configurar_pagina():
        """Configura as opções da página Streamlit."""
        st.set_page_config(
            layout="wide",
            page_title="Dashboard Preditivo de Ativos",
            page_icon="📊"
        )

    def _inicializar_sidebar(self):
        """Inicializa a barra lateral com controles."""
        st.sidebar.title("🎯 Painel de Controle")

        self.tickers_disponiveis = [
            "PETR4.SA", "VALE3.SA", "ITSA4.SA",
            "TAEE11.SA", "BBSE3.SA", "ABEV3.SA"
        ]

        self.ticker_selecionado = st.sidebar.selectbox(
            "Selecione o Ativo",
            self.tickers_disponiveis
        )

        self.analisar_btn = st.sidebar.button(
            "🔍 Analisar Ativo",
            type="primary"
        )

    @staticmethod
    @st.cache_resource
    def _carregar_modelo(ticker: str) -> Optional[Any]:
        """
        Carrega modelo pré-treinado do disco.

        Args:
            ticker: Símbolo do ativo

        Returns:
            Modelo carregado ou None se não encontrado
        """
        caminho_modelo = f"modelos_treinados/modelo_{ticker}.joblib"

        if os.path.exists(caminho_modelo):
            try:
                return load(caminho_modelo)
            except Exception as e:
                st.error(f"Erro ao carregar modelo: {e}")
                return None
        return None

    def _exibir_cabecalho(self):
        """Exibe cabeçalho da página."""
        st.title("📈 Dashboard de Análise Preditiva com Machine Learning")
        st.markdown(f"**Analisando o ativo:** `{self.ticker_selecionado}`")
        st.divider()

    @staticmethod
    def _exibir_previsao(previsao: Dict[str, Any]):
        """Exibe resultados da previsão."""
        st.header("🎯 Previsão para o Próximo Dia de Pregão")

        col1, col2, col3 = st.columns(3)

        direcao = "📈 ALTA" if previsao['predicao'] == 1 else "📉 BAIXA"
        recomendacao = "✅ OPERAR" if previsao['should_operate'] else "⏸️ AGUARDAR"

        col1.metric(
            "Recomendação",
            recomendacao,
            help="Decisão de operar baseada na confiança do modelo"
        )

        col2.metric(
            "Direção Prevista",
            direcao,
            help="Previsão de alta ou baixa para o próximo pregão"
        )

        col3.metric(
            "Confiança do Modelo",
            f"{previsao['probabilidade']:.1%}",
            help=f"O modelo só recomenda operar se a confiança for maior que {previsao['limiar_confianca']:.1%}"
        )

    @staticmethod
    def _criar_grafico_precos_sinais(precos: pd.Series, df_sinais: pd.DataFrame,
                                     limiar_confianca: float) -> go.Figure:
        """Cria gráfico de preços com sinais de operação."""
        fig = go.Figure()

        # Linha de preços
        fig.add_trace(go.Scatter(
            x=precos.index,
            y=precos,
            mode='lines',
            name='Preço de Fechamento',
            line=dict(color='#1f77b4', width=2)
        ))

        # Sinais de operação
        sinais_operar = df_sinais[
            (df_sinais['pred'] == 1) &
            (df_sinais['proba'] >= limiar_confianca)
            ]

        if not sinais_operar.empty:
            fig.add_trace(go.Scatter(
                x=sinais_operar.index,
                y=sinais_operar['preco'],
                mode='markers',
                marker=dict(
                    color='limegreen',
                    size=10,
                    symbol='triangle-up',
                    line=dict(width=1, color='darkgreen')
                ),
                name='Sinal de Operação',
                hovertemplate='<b>Data:</b> %{x}<br><b>Preço:</b> R$ %{y:.2f}<br>'
            ))

        fig.update_layout(
            title_text='📊 Preços Históricos vs. Sinais de Operação do Modelo',
            xaxis_title='Data',
            yaxis_title='Preço (R$)',
            hovermode='x unified',
            showlegend=True
        )

        return fig

    @staticmethod
    def _criar_grafico_equidade(curva_equidade: list, datas: pd.DatetimeIndex) -> go.Figure:
        """Cria gráfico da curva de equidade."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=datas,
            y=curva_equidade,
            mode='lines',
            name='Crescimento do Capital',
            line=dict(color='#2ca02c', width=3),
            hovertemplate='<b>Data:</b> %{x}<br><b>Capital:</b> %{y:.2f}x<br>'
        ))

        fig.update_layout(
            title_text='💰 Curva de Equidade (Backtest)',
            xaxis_title='Data',
            yaxis_title='Capital Relativo',
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def _exibir_metricas_backtest(metricas: Dict[str, Any]):
        """Exibe métricas de performance do backtest."""
        st.subheader("📊 Métricas de Performance do Backtest")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Retorno Total",
            f"{metricas['retorno_total']:.2%}",
            help="Retorno acumulado no período"
        )

        col2.metric(
            "Sharpe Ratio",
            f"{metricas['sharpe']:.2f}",
            help="Risco-retorno ajustado (anualizado)"
        )

        col3.metric(
            "Max Drawdown",
            f"{metricas['max_drawdown']:.2%}",
            help="Maior perda em relação ao pico"
        )

        col4.metric(
            "Nº de Trades",
            f"{metricas['trades']}",
            help="Total de operações realizadas"
        )

    @staticmethod
    def _exibir_explicabilidade(modelo):
        """Exibe informações sobre como o modelo toma decisões."""
        with st.expander("🔍 Como o modelo toma as decisões?", expanded=False):
            st.write("""
            O modelo foi treinado para identificar padrões históricos nos dados de mercado 
            e toma decisões com base nas seguintes features consideradas mais importantes:
            """)

            df_features = pd.DataFrame(
                modelo.features_selecionadas,
                columns=['Features Selecionadas']
            )

            st.dataframe(
                df_features,
                use_container_width=True,
                hide_index=True
            )

            st.info("""
            💡 **Nota:** O modelo usa ensemble learning combinando múltiplos algoritmos 
            e só opera quando a confiança na previsão é suficientemente alta.
            """)

    def executar_analise(self):
        """Executa análise completa do ativo selecionado."""
        if not self.analisar_btn:
            return

        self._exibir_cabecalho()

        # Carregar modelo
        modelo = self._carregar_modelo(self.ticker_selecionado)

        if modelo is None:
            st.error(
                f"❌ Modelo para {self.ticker_selecionado} não encontrado. "
                f"Execute o script 'train.py' primeiro."
            )
            return

        # Inicializar utilitários
        loader = DataLoader()
        feature_engineer = FeatureEngineer()
        risk_analyzer = RiskAnalyzer()

        # Obter dados e fazer previsão
        with st.spinner("📥 Buscando dados e fazendo nova previsão..."):
            try:
                df_ticker, df_ibov = loader.baixar_dados_yf(
                    self.ticker_selecionado,
                    periodo="3y"
                )

                X_full, y_full, precos_full = feature_engineer.preparar_dataset_classificacao(
                    df_ticker,
                    df_ibov
                )

                X_novo = X_full.tail(1)
                previsao = modelo.prever_direcao(X_novo)

            except Exception as e:
                st.error(f"❌ Erro ao processar dados: {e}")
                return

        # Exibir previsão
        self._exibir_previsao(previsao)
        st.divider()

        # Análise de performance histórica
        with st.spinner("📊 Calculando performance histórica..."):
            try:
                df_sinais = modelo.prever_e_gerar_sinais(
                    X_full,
                    precos_full,
                    retornar_dataframe=True
                )

                backtest_info = risk_analyzer.backtest_sinais(df_sinais)

            except Exception as e:
                st.error(f"❌ Erro ao calcular performance: {e}")
                return

        st.header("📈 Performance Histórica do Modelo (Backtest)")

        # Gráfico de preços e sinais
        fig_precos = self._criar_grafico_precos_sinais(
            precos_full,
            df_sinais,
            modelo.confidence_operar
        )
        st.plotly_chart(fig_precos, use_container_width=True)

        # Gráfico de curva de equidade
        if 'equity_curve' in backtest_info and backtest_info['equity_curve']:
            fig_equity = self._criar_grafico_equidade(
                backtest_info['equity_curve'],
                df_sinais.index[:len(backtest_info['equity_curve'])]
            )
            st.plotly_chart(fig_equity, use_container_width=True)

        # Métricas de backtest
        self._exibir_metricas_backtest(backtest_info)
        st.divider()

        # Explicabilidade do modelo
        self._exibir_explicabilidade(modelo)


dashboard = DashboardTrading()
dashboard.executar_analise()
