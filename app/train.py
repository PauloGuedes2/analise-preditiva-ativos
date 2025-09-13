import os

from joblib import dump

from src.data.data_loader import DataLoaderRefinado
from src.models.classification import ClassificacaoFinalRefinado
from src.models.feature_engineer import FeatureEngineerRefinado


def treinar_e_salvar_modelos():
    tickers = ["PETR4.SA", "VALE3.SA", "ITSA4.SA", "TAEE11.SA", "BBSE3.SA", "ABEV3.SA"]
    modelos_dir = "modelos_treinados"
    os.makedirs(modelos_dir, exist_ok=True)

    for ticker in tickers:
        print(f"\n🚀 Treinando modelo para {ticker}...")
        try:
            # Carregar dados
            loader = DataLoaderRefinado()
            # AGORA RECEBE DOIS DATAFRAMES:
            df_ohlc, df_ibov = loader.baixar_dados_yf(ticker, periodo="3y")

            if len(df_ohlc) < 100:
                print(f"⚠️ Dados insuficientes para {ticker}")
                continue

            # Engenharia de Features
            fe = FeatureEngineerRefinado()
            # PASSA OS DOIS DATAFRAMES AQUI:
            X, y, precos = fe.preparar_dataset_classificacao(df_ohlc, df_ibov)

            # Treinamento do Modelo
            model = ClassificacaoFinalRefinado(n_features=20, random_state=42)
            model.treinar(X, y, precos, n_splits=4, purge_days=2)
            # Salvar o objeto do modelo treinado
            caminho_modelo = os.path.join(modelos_dir, f"modelo_{ticker}.joblib")
            dump(model, caminho_modelo)
            print(f"✅ Modelo para {ticker} salvo em: {caminho_modelo}")

        except Exception as e:
            print(f"❌ Erro ao treinar {ticker}: {e}")


if __name__ == "__main__":
    treinar_e_salvar_modelos()
