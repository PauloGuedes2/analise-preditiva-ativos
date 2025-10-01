import atexit
import os
import time
from datetime import datetime
from typing import Any, Tuple

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
    def _carregar_modelo(_self, ticker: str) -> Tuple[Any, Any]:
        """
        Carrega o modelo treinado do disco e sua data de treinamento.
        Retorna uma tupla (modelo, data_treinamento).
        """
        caminho = os.path.join(Params.PATH_MODELOS, f"modelo_{ticker}.joblib")
        if os.path.exists(caminho):
            try:
                modelo = load(caminho)

                # Lógica de fallback para data de treinamento
                if hasattr(modelo, 'data_treinamento'):
                    # Usa a data salva no modelo (para modelos novos)
                    data = modelo.data_treinamento
                else:
                    # Usa a data de modificação do arquivo (para modelos antigos)
                    timestamp = os.path.getmtime(caminho)
                    data = datetime.fromtimestamp(timestamp)

                return modelo, data

            except Exception as e:
                st.error(f"Erro ao carregar o modelo '{ticker}': {e}")
                return None, None
        return None, None

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

        data_base_analise = X_untruncated.index[-1] if not X_untruncated.empty else None
        data_alvo_previsao = (data_base_analise + pd.tseries.offsets.BDay(1)) if data_base_analise else None

        return {
            "df_ticker": df_ticker, "df_ibov": df_ibov, "X_full": X_full,
            "y_full": y_full, "precos_full": precos_full, "previsao": previsao,
            "t1": t1,
            "data_base_analise": data_base_analise,
            "data_alvo_previsao": data_alvo_previsao,
            "X_untruncated": X_untruncated
        }

    def _gerar_validacao_recente(self, dados: dict) -> tuple[list, dict]:
        """
        Gera a validação de performance para os últimos N dias, garantindo que
        as datas sejam contínuas até o dia da análise.
        """
        num_dias = Params.UI_VALIDATION_DAYS
        X_untruncated = dados["X_untruncated"]
        y_full = dados["y_full"]
        precos_full = dados["precos_full"]

        if len(X_untruncated) < num_dias:
            return [], {}

        resultados_validacao = []
        acertos_retornos = []
        num_oportunidades, num_acertos, num_decisoes_corretas = 0, 0, 0

        dias_avaliados = 0
        dias_para_validar = min(num_dias - 1, len(X_untruncated) - 1)

        for dia in X_untruncated.index[-(dias_para_validar + 1):-1]:
            dados_dia = X_untruncated.loc[dia:dia]
            previsao_hist = self.modelo_carregado.prever_direcao(dados_dia, self.ticker_selecionado)

            resultado_real_tb = y_full.get(dia, "N/A")

            try:
                idx_atual = precos_full.index.get_loc(dia)
                variacao_real = (precos_full.iloc[idx_atual + 1] / precos_full.iloc[
                    idx_atual]) - 1 if idx_atual + 1 < len(
                    precos_full) else np.nan
            except (KeyError, IndexError):
                variacao_real = np.nan

            performance_str = "⚪️ Neutro"
            sinal_foi_oportunidade = previsao_hist['should_operate']

            if not np.isnan(variacao_real):
                dias_avaliados += 1
                if sinal_foi_oportunidade:
                    num_oportunidades += 1
                    if variacao_real > 0:
                        performance_str = "✅ Acerto"
                        num_acertos += 1
                        num_decisoes_corretas += 1
                        acertos_retornos.append(variacao_real)
                    else:
                        performance_str = "❌ Erro"
                else:
                    if variacao_real > 0:
                        performance_str = "⚪️ Alta não Sinalizada"
                    else:
                        performance_str = "✅ Evitou Perda"
                        num_decisoes_corretas += 1

            resultados_validacao.append({
                "Data": dia.strftime('%d/%m/%Y'),
                "Sinal do Modelo": "🟢 OPORTUNIDADE" if sinal_foi_oportunidade else "🟡 OBSERVAR",
                "Probabilidade de Alta": previsao_hist['probabilidade'],
                "Resultado Real (Tripla Barreira)": resultado_real_tb,
                "Variação Diária Real": variacao_real,
                "Performance": performance_str
            })

        previsao_atual = dados['previsao']
        data_alvo = dados.get("data_alvo_previsao")
        data_alvo_formatada = data_alvo.strftime('%d/%m/%Y') if data_alvo else "Data Indisponível"

        resultados_validacao.append({
            "Data": data_alvo_formatada,
            "Sinal do Modelo": "🟢 OPORTUNIDADE" if previsao_atual['should_operate'] else "🟡 OBSERVAR",
            "Probabilidade de Alta": previsao_atual['probabilidade'],
            "Resultado Real (Tripla Barreira)": "⏳ Aguardando",
            "Variação Diária Real": np.nan,
            "Performance": "⏳ Em Acompanhamento"
        })

        metricas = {
            'taxa_acerto': (num_acertos / num_oportunidades) if num_oportunidades > 0 else 0,
            'retorno_medio_acertos': np.mean(acertos_retornos) if acertos_retornos else 0,
            'assertividade_geral': (num_decisoes_corretas / dias_avaliados) if dias_avaliados > 0 else 0,
            'num_oportunidades_recente': num_oportunidades
        }
        return resultados_validacao, metricas

    def executar(self):
        """Orquestra o fluxo principal da aplicação."""
        if not self.ticker_selecionado:
            self.view.render_tela_boas_vindas()
            return

        self.modelo_carregado, data_treinamento = self._carregar_modelo(self.ticker_selecionado)

        if self.modelo_carregado is None:
            st.error(f"O modelo para {self.ticker_selecionado} não pôde ser carregado.")
            st.stop()

        dados = self._processar_dados_e_previsao(self.ticker_selecionado, self.modelo_carregado)

        validacao_recente, metricas_validacao = self._gerar_validacao_recente(dados)

        self.view.render_main_layout(
            ticker=self.ticker_selecionado,
            modelo=self.modelo_carregado,
            dados=dados,
            validacao_recente=validacao_recente,
            metricas_validacao=metricas_validacao,
            data_treinamento=data_treinamento
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
