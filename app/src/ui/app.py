import atexit
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from src.config.params import Params
from src.data.data_loader import DataLoader
from src.data.data_updater import data_updater
from src.models.feature_engineer import FeatureEngineer
from src.ui.dashboard_view import DashboardView

st.set_page_config(layout="wide", page_title="Análise Preditiva de Ativos", page_icon="📈")


class DashboardTrading:
    """Controlador do Dashboard Streamlit para Análise Preditiva de Ativos."""

    def __init__(self):
        """Inicializa o controlador, os serviços e a visão."""
        self.modelo_carregado: Any = None
        self.ticker_selecionado: str = ""
        self.view = DashboardView(st)

        # Inicia a atualização de dados em background
        data_updater.iniciar_atualizacao_automatica(tickers=Params.TICKERS)

        self.ticker_selecionado = self._inicializar_sidebar()

    def _inicializar_sidebar(self) -> str:
        """Configura a barra lateral e retorna o ticker selecionado."""
        with st.sidebar:
            st.markdown("## 📈 Análise Preditiva")
            st.markdown("---")

            # Carrega a lista de modelos disponíveis
            modelos_disponiveis = sorted(
                [f.replace('modelo_', '').replace('.joblib', '') for f in os.listdir(Params.PATH_MODELOS) if
                 f.endswith('.joblib')]
            )

            if not modelos_disponiveis:
                st.warning("Nenhum modelo treinado foi encontrado.")
                st.stop()

            ticker = st.selectbox("Selecione o Ativo:", modelos_disponiveis, key="ticker_selector")
            st.markdown("---")

            with st.expander("Manutenção e Ajuda"):
                if st.button("🔄 Forçar Reset dos Dados", use_container_width=True,
                             help="Apaga o banco de dados local para forçar o download de dados novos na próxima análise."):
                    self._forcar_download_dados()
        return ticker

    @st.cache_resource(ttl=3600, show_spinner="Carregando modelo...")
    def _carregar_modelo(_self, ticker: str) -> Any:
        """Carrega o modelo treinado do disco."""
        caminho = os.path.join(Params.PATH_MODELOS, f"modelo_{ticker}.joblib")
        if os.path.exists(caminho):
            try:
                return load(caminho)
            except Exception as e:
                st.error(f"Erro ao carregar o modelo '{ticker}': {e}")
                return None
        return None

    @st.cache_data(show_spinner="Processando dados do mercado...")
    def _processar_dados_e_previsao(_self, ticker: str, _modelo: Any) -> dict:
        """
        Orquestra o download, processamento de dados e geração de previsão.
        """
        loader = DataLoader()
        feature_engineer = FeatureEngineer()

        try:
            df_ticker, df_ibov = loader.baixar_dados_yf(ticker)
        except Exception as e:
            st.warning(f"**Aviso:** Falha ao baixar dados ({e}). Usando a última versão salva no banco de dados local.")
            df_ticker = loader.carregar_do_bd(ticker)
            df_ibov = loader.carregar_do_bd('^BVSP')

        if df_ticker.empty:
            st.error(f"Não foi possível carregar dados para {ticker}.")
            st.stop()

        X_full, y_full, precos_full, t1, X_untruncated = feature_engineer.preparar_dataset(
            df_ticker, df_ibov, ticker
        )

        previsao = _modelo.prever_direcao(X_untruncated.tail(1), ticker)

        return {
            "df_ticker": df_ticker, "df_ibov": df_ibov, "X_full": X_full,
            "y_full": y_full, "precos_full": precos_full, "previsao": previsao,
            "t1": t1
        }

    def _gerar_validacao_recente(self, X_full: pd.DataFrame, y_full: pd.Series, precos_full: pd.Series) -> tuple[
        list, dict]:
        """Gera previsões para os últimos N dias e calcula métricas de performance."""
        num_dias = Params.UI_VALIDATION_DAYS
        if len(X_full) < num_dias or len(precos_full) <= num_dias:
            return [], {}

        resultados_validacao = []
        acertos_retornos = []
        num_oportunidades, num_acertos = 0, 0

        for dia in X_full.index[-num_dias:]:
            dados_dia = X_full.loc[dia:dia]
            previsao = self.modelo_carregado.prever_direcao(dados_dia, self.ticker_selecionado)
            resultado_real = y_full.loc[dia]
            acertou = (previsao['should_operate'] and resultado_real == 1)

            try:
                idx_atual = precos_full.index.get_loc(dia)
                variacao_real = (precos_full.iloc[idx_atual + 1] / precos_full.iloc[
                    idx_atual]) - 1 if idx_atual + 1 < len(precos_full) else np.nan
            except (KeyError, IndexError):
                variacao_real = np.nan

            if previsao['should_operate']:
                num_oportunidades += 1
                if acertou:
                    num_acertos += 1
                    if not np.isnan(variacao_real): acertos_retornos.append(variacao_real)

            resultados_validacao.append({
                "Data": dia.strftime('%d/%m/%Y'),
                "Sinal do Modelo": "🟢 OPORTUNIDADE" if previsao['should_operate'] else "🟡 OBSERVAR",
                "Confiança do Modelo": previsao['probabilidade'],
                "Resultado Real (Tripla Barreira)": resultado_real,
                "Variação Diária Real": variacao_real,
                "Performance": "✅ ACERTOU" if acertou else ("❌ ERROU" if previsao['should_operate'] else "⚪️ Neutro")
            })

        metricas = {
            'taxa_acerto': (num_acertos / num_oportunidades) if num_oportunidades > 0 else 0,
            'retorno_medio_acertos': np.mean(acertos_retornos) if acertos_retornos else 0
        }
        return resultados_validacao, metricas

    def executar(self):
        """Orquestra o fluxo principal da aplicação."""
        if not self.ticker_selecionado:
            self.view.render_tela_boas_vindas()
            return

        self.modelo_carregado = self._carregar_modelo(self.ticker_selecionado)

        if self.modelo_carregado is None:
            st.error(f"O modelo para {self.ticker_selecionado} não pôde ser carregado.")
            st.stop()

        dados = self._processar_dados_e_previsao(self.ticker_selecionado, self.modelo_carregado)

        validacao_recente, metricas_validacao = self._gerar_validacao_recente(
            dados["X_full"], dados["y_full"], dados["precos_full"]
        )

        self.view.render_main_layout(
            ticker=self.ticker_selecionado,
            modelo=self.modelo_carregado,
            dados=dados,
            validacao_recente=validacao_recente,
            metricas_validacao=metricas_validacao
        )

    @staticmethod
    def _forcar_download_dados():
        """Gerencia o estado e sistema de arquivos para forçar o download de dados."""
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
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")

    @staticmethod
    @atexit.register
    def parar_servicos():
        """Registra a parada de serviços ao sair da aplicação."""
        data_updater.parar_atualizacao()


if __name__ == "__main__":
    dashboard = DashboardTrading()
    dashboard.executar()
