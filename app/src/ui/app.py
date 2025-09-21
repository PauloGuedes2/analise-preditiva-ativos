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

st.set_page_config(layout="wide", page_title="Dashboard Preditivo de Ativos", page_icon="📊")


class DashboardTrading:
    """Dashboard interativo para análise preditiva de ativos."""

    def __init__(self):
        self.modelo_carregado = None
        self._inicializar_sidebar()

    def _inicializar_sidebar(self):
        """Inicializa a barra lateral com os controles do usuário."""
        with st.sidebar:
            st.title("🎯 Painel de Controle")
            self.ticker_selecionado = st.selectbox("Selecione o Ativo", Params.TICKERS)
            self.analisar_btn = st.button("🔍 Analisar Ativo", type="primary", use_container_width=True)

    @st.cache_resource
    def _carregar_modelo(_self, ticker: str) -> Optional[Any]:
        """Carrega o modelo treinado do disco. A anotação _self é para usar st.cache_resource em um método."""
        caminho = os.path.join(Params.PATH_MODELOS, f"modelo_{ticker}.joblib")
        if os.path.exists(caminho):
            try:
                return load(caminho)
            except Exception as e:
                st.error(f"Erro ao carregar o modelo: {e}")
        return None

    def _exibir_previsao(self, previsao: Dict[str, Any]):
        """Mostra os resultados da previsão mais recente."""
        st.header("🎯 Previsão para o Próximo Dia de Pregão")

        if previsao.get('status') != 'sucesso':
            st.error(f"Não foi possível gerar a previsão: {previsao.get('status')}")
            return

        cols = st.columns(3)
        direcao = "📈 ALTA" if previsao['predicao'] == 1 else "📉 BAIXA"
        recomendacao = "✅ OPERAR" if previsao['should_operate'] else "⏸️ AGUARDAR"

        cols[0].metric("Recomendação", recomendacao,
                       help="Decisão baseada na confiança do modelo ser maior que o limiar definido.")
        cols[1].metric("Direção Prevista", direcao, help="Previsão de alta ou baixa para o próximo pregão.")
        cols[2].metric("Confiança do Modelo", f"{previsao['probabilidade']:.1%}",
                       help=f"A recomendação de operar é ativada se esta confiança for maior que o Threshold Calibrado de {previsao['threshold_operacional']:.1%}")

    def _criar_grafico_precos_sinais(self, df_sinais: pd.DataFrame, precos: pd.Series) -> go.Figure:
        """Cria um gráfico de preços com os sinais de operação."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=precos.index, y=precos, mode='lines', name='Preço de Fechamento',
                                 line=dict(color='#1f77b4', width=2)))

        sinais_operar = df_sinais[df_sinais['sinal'] == 1]
        if not sinais_operar.empty:
            fig.add_trace(go.Scatter(
                x=sinais_operar.index, y=sinais_operar['preco'], mode='markers',
                marker=dict(color='limegreen', size=10, symbol='triangle-up', line=dict(width=1, color='darkgreen')),
                name='Sinal de Compra'
            ))

        fig.update_layout(title_text='📊 Preços Históricos vs. Sinais de Operação', xaxis_title='Data',
                          yaxis_title='Preço (R$)', hovermode='x unified')
        return fig

    def _criar_grafico_equidade(self, backtest_info: Dict[str, Any]) -> go.Figure:
        """Cria o gráfico da curva de equidade do backtest."""
        fig = go.Figure()
        curva_equidade = backtest_info.get('equity_curve', [])
        if curva_equidade:
            # O número de pontos na curva de equidade corresponde ao número de trades.
            # Precisamos de um eixo x apropriado.
            datas_trades = pd.to_datetime(
                np.linspace(pd.Timestamp('2022-01-01').value, pd.Timestamp('2024-01-01').value, len(curva_equidade)))

            fig.add_trace(go.Scatter(
                x=datas_trades, y=curva_equidade, mode='lines', name='Crescimento do Capital',
                line=dict(color='#2ca02c', width=3)
            ))
        fig.update_layout(title_text='💰 Curva de Equidade (Backtest)', xaxis_title='Trades (tempo)',
                          yaxis_title='Capital Relativo', hovermode='x unified')
        return fig

    def _exibir_metricas_backtest(self, metricas: Dict[str, Any]):
        """Mostra as principais métricas de performance do backtest."""
        st.subheader("📊 Métricas de Performance do Backtest")
        cols = st.columns(5)
        cols[0].metric("Retorno Total", f"{metricas.get('retorno_total', 0):.2%}")
        cols[1].metric("Sharpe Ratio", f"{metricas.get('sharpe', 0):.2f}")
        cols[2].metric("Max Drawdown", f"{metricas.get('max_drawdown', 0):.2%}")
        cols[3].metric("Taxa de Acerto", f"{metricas.get('win_rate', 0):.2%}")
        cols[4].metric("Nº de Trades", f"{metricas.get('trades', 0)}")

    def _exibir_explicabilidade(self):
        """Mostra as features mais importantes para o modelo."""
        with st.expander("🔍 Como o modelo toma as decisões?", expanded=False):
            st.write(
                "O modelo foi treinado para identificar padrões nos dados de mercado usando um algoritmo LightGBM. As decisões são baseadas nas seguintes features, consideradas as mais importantes durante o treinamento:")
            df_features = pd.DataFrame(self.modelo_carregado.features_selecionadas,
                                       columns=['Features Mais Importantes'])
            st.dataframe(df_features, use_container_width=True, hide_index=True)
            st.info(
                "💡 O modelo usa um ensemble (Stacking) e só recomenda operar quando a confiança na previsão de alta é suficientemente elevada.")

    def _exibir_metricas_validadas(self, metricas: Dict[str, Any]):
        """Mostra as métricas de performance da Validação Walk-Forward (Out-of-Sample)."""
        st.subheader("📊 Performance Real Estimada (Validação Walk-Forward)")
        st.info(
            "Estas são as métricas mais realistas, calculadas em dados não vistos ('out-of-sample') durante o treinamento.")

        cols = st.columns(3)
        cols[0].metric("Sharpe Ratio Médio", f"{metricas.get('sharpe_medio', 0):.2f}")
        cols[1].metric("Média de Trades por Fold", f"{metricas.get('trades_medio', 0):.1f}")
        cols[2].metric("F1-Score Macro Médio", f"{metricas.get('f1_macro_medio', 0):.2%}")

    def executar_analise(self):
        """Orquestra a análise completa quando o botão é clicado."""
        if not self.analisar_btn:
            st.info("Selecione um ativo e clique em 'Analisar' para começar.")
            return

        st.title(f"📈 Análise Preditiva para {self.ticker_selecionado}")
        self.modelo_carregado = self._carregar_modelo(self.ticker_selecionado)

        if self.modelo_carregado is None:
            st.error(f"Modelo para {self.ticker_selecionado} não encontrado. Execute o script 'train.py' primeiro.")
            return

        with st.spinner("Buscando dados e gerando nova previsão..."):
            try:
                loader = DataLoader()
                feature_engineer = FeatureEngineer()
                df_ticker, df_ibov = loader.baixar_dados_yf(self.ticker_selecionado)
                # Passar o ticker aqui
                X_full, _, precos_full, _ = feature_engineer.preparar_dataset(df_ticker, df_ibov,
                                                                              self.ticker_selecionado)
                previsao = self.modelo_carregado.prever_direcao(X_full.tail(1), self.ticker_selecionado)
            except Exception as e:
                st.error(f"Erro ao processar dados: {e}")
                return

            self._exibir_previsao(previsao)

            if hasattr(self.modelo_carregado, 'wfv_metrics') and self.modelo_carregado.wfv_metrics:
                self._exibir_metricas_validadas(self.modelo_carregado.wfv_metrics)

            st.divider()

            st.header("📈 Simulação Histórica (Backtest em Dados Completos)")
            st.warning("Atenção: A simulação abaixo é 'in-sample' (executada nos mesmos dados usados para treinar o modelo final). Seus resultados são otimistas e servem para visualização, mas a performance real é a estimada pela Validação Walk-Forward acima.")

            with st.spinner("Calculando performance histórica..."):
                try:
                    risk_analyzer = RiskAnalyzer()

                    df_sinais = self.modelo_carregado.prever_e_gerar_sinais(X_full, precos_full, self.ticker_selecionado)
                    backtest_info = risk_analyzer.backtest_sinais(df_sinais)
                except Exception as e:
                    st.error(f"Erro ao calcular performance: {e}")
                    return

            fig_precos = self._criar_grafico_precos_sinais(df_sinais, precos_full)

            st.plotly_chart(fig_precos, use_container_width=True)

            fig_equity = self._criar_grafico_equidade(backtest_info)
            st.plotly_chart(fig_equity, use_container_width=True)

            self._exibir_metricas_backtest(backtest_info)
            st.divider()

            self._exibir_explicabilidade()


if __name__ == "__main__":
    dashboard = DashboardTrading()
    dashboard.executar_analise()
