import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load

from src.data.data_loader import DataLoaderRefinado
from src.models.feature_engineer import FeatureEngineerRefinado
from src.utils.risk_analyzer import RiskAnalyzerRefinado

# --- Configurações da Página ---
st.set_page_config(layout="wide", page_title="Dashboard Preditivo de Ativos")


# --- Funções de Cache ---
@st.cache_resource
def carregar_modelo(ticker):
    """Carrega um modelo pré-treinado do disco."""
    caminho_modelo = f"modelos_treinados/modelo_{ticker}.joblib"
    if os.path.exists(caminho_modelo):
        modelo = load(caminho_modelo)
        return modelo
    return None


# --- Interface do Usuário (Sidebar) ---
st.sidebar.title("Painel de Controle")
# Lista de tickers para os quais você treinou modelos
TICKERS_DISPONIVEIS = ["PETR4.SA", "VALE3.SA", "ITSA4.SA", "TAEE11.SA", "BBSE3.SA", "ABEV3.SA"]
ticker_selecionado = st.sidebar.selectbox("Selecione o Ativo", TICKERS_DISPONIVEIS)
analisar_btn = st.sidebar.button("Analisar Ativo")

# --- Área Principal ---
st.title("Dashboard de Análise Preditiva com Machine Learning")
st.markdown(f"Analisando o ativo: **{ticker_selecionado}**")

if analisar_btn:
    modelo = carregar_modelo(ticker_selecionado)

    if modelo is None:
        st.error(f"Modelo para {ticker_selecionado} não encontrado. Execute o script 'train.py' primeiro.")
    else:
        # Ferramentas auxiliares
        loader = DataLoaderRefinado()
        fe = FeatureEngineerRefinado()
        ra = RiskAnalyzerRefinado()

        # 1. Obter dados e fazer a previsão mais recente
        with st.spinner("Buscando dados e fazendo nova previsão..."):
            df_ticker, df_ibov = loader.baixar_dados_yf(ticker_selecionado, periodo="3y")
            X_full, y_full, precos_full = fe.preparar_dataset_classificacao(df_ticker, df_ibov)

            X_novo = X_full.tail(1)
            previsao = modelo.prever_direcao(X_novo)

        st.header("Previsão para o Próximo Dia de Pregão")
        col1, col2, col3 = st.columns(3)
        direcao = "📈 ALTA" if previsao['predicao'] == 1 else "📉 BAIXA"
        recomendacao = "✅ OPERAR" if previsao['should_operate'] else "⏸️ AGUARDAR"

        col1.metric("Recomendação", recomendacao)
        col2.metric("Direção Prevista", direcao)
        col3.metric("Confiança do Modelo", f"{previsao['probabilidade']:.1%}",
                    help=f"O modelo só recomenda operar se a confiança for maior que {previsao['limiar_confianca']:.1%}")

        # 2. Análise de Performance Histórica (Backtest)
        with st.spinner("Calculando performance histórica..."):
            df_sinais = modelo.prever_e_gerar_sinais(X_full, precos_full, retornar_dataframe=True)
            backtest_info = ra.backtest_sinais(df_sinais)

        st.header("Performance Histórica do Modelo (Backtest)")

        # Gráfico 1: Preços x Sinais
        fig_precos = go.Figure()
        fig_precos.add_trace(go.Scatter(x=precos_full.index, y=precos_full, mode='lines', name='Preço de Fechamento'))
        sinais_operar = df_sinais[(df_sinais['pred'] == 1) & (df_sinais['proba'] >= modelo.confidence_operar)]
        fig_precos.add_trace(go.Scatter(x=sinais_operar.index, y=sinais_operar['preco'], mode='markers',
                                        marker=dict(color='limegreen', size=10, symbol='triangle-up'),
                                        name='Sinal de Operação'))
        fig_precos.update_layout(title_text='Preços Históricos vs. Sinais de Operação do Modelo', xaxis_title='Data',
                                 yaxis_title='Preço (R$)')
        st.plotly_chart(fig_precos, use_container_width=True)

        # Gráfico 2: Curva de Equidade
        if 'equity_curve' in backtest_info:
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(x=sinais_operar.index, y=backtest_info['equity_curve'], mode='lines',
                                            name='Crescimento do Capital'))
            fig_equity.update_layout(title_text='Curva de Equidade (Backtest)', xaxis_title='Data',
                                     yaxis_title='Capital Relativo')
            st.plotly_chart(fig_equity, use_container_width=True)

        # Métricas do Backtest
        st.subheader("Métricas de Performance do Backtest")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Retorno Total", f"{backtest_info['retorno_total']:.2%}")
        b2.metric("Sharpe Ratio (Anualizado)", f"{backtest_info['sharpe']:.2f}")
        b3.metric("Max Drawdown", f"{backtest_info['max_drawdown']:.2%}")
        b4.metric("Nº de Trades", f"{backtest_info['trades']}")

        # 3. Explicabilidade do Modelo
        with st.expander("🔍 Como o modelo toma as decisões?"):
            st.write(
                "O modelo foi treinado para tomar decisões com base nas seguintes features, consideradas as mais importantes:")
            st.dataframe(pd.DataFrame(modelo.features_selecionadas, columns=['Features Selecionadas']))
