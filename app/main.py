"""
Script principal - fluxo completo:
1. Baixar dados via yfinance
2. Salvar no banco
3. Preparar dataset
4. Treinar modelo
5. Gerar previsão para o próximo dia
"""

from src.data.data_loader import DataLoaderRefinado
from src.models.classification import ClassificacaoFinalRefinado
from src.models.feature_engineer import FeatureEngineerRefinado


def main():
    tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA"]

    for ticker in tickers:
        print(f"\n🎯 Testando {ticker}")
        print("=" * 50)

        loader = DataLoaderRefinado()
        df_ohlc = loader.baixar_dados_yf(ticker, periodo="3y", intervalo="1d")

        if len(df_ohlc) < 100:
            print(f"⚠️  Dados insuficientes para {ticker}")
            continue

        fe = FeatureEngineerRefinado()
        X, y, precos = fe.preparar_dataset_classificacao(df_ohlc)

        print(f"📊 Dados: {len(df_ohlc)} registros, Features: {X.shape}")
        print(f"📈 Balanceamento: {y.value_counts(normalize=True).to_dict()}")

        try:
            model = ClassificacaoFinalRefinado(n_features=20, random_state=42)
            meta = model.treinar(X, y, precos, n_splits=4, purge_days=2)

            # Previsão - verificar se o método existe
            if hasattr(model, 'prever_direcao'):
                X_novo = X.tail(1)
                resultado = model.prever_direcao(X_novo)
                print(f"🔮 Previsão: {resultado}")
            else:
                print("⚠️  Método prever_direcao não disponível")

            print(f"💰 Backtest: {meta['backtest']}")

        except Exception as e:
            print(f"❌ Erro no treinamento de {ticker}: {e}")
            import traceback
            traceback.print_exc()


# def monitorar_mercado():
#     """Monitora múltiplos ativos em tempo real"""
#     tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "B3SA3.SA", "WEGE3.SA"]
#
#     print("📡 Iniciando monitoramento do mercado...")
#     print("=" * 60)
#
#     resultados = []
#
#     for ticker in tickers:
#         try:
#             loader = DataLoaderRefinado()
#             df_ohlc = loader.baixar_dados_yf(ticker, periodo="1y", intervalo="1d")  # Menos dados para velocidade
#
#             if len(df_ohlc) < 50:
#                 continue
#
#             fe = FeatureEngineerRefinado()
#             X, y, precos = fe.preparar_dataset_classificacao(df_ohlc)
#
#             # Usar modelo pré-treinado ou treinar rápido
#             model = ClassificacaoFinalRefinado(n_features=15)
#             model.treinar(X, y, precos, n_splits=3, purge_days=1)
#
#             # Previsão
#             X_novo = X.tail(1)
#             resultado = model.prever_direcao(X_novo)
#
#             # Obter o último preço como float
#             ultimo_preco = float(df_ohlc['Close'].iloc[-1])
#
#             resultados.append({
#                 'ticker': ticker,
#                 'previsao': resultado['predicao'],
#                 'probabilidade': resultado['probabilidade'],
#                 'operar': resultado['should_operate'],
#                 'ultimo_preco': ultimo_preco
#             })
#
#         except Exception as e:
#             print(f"⚠️  Erro em {ticker}: {e}")
#
#     # Exibir resultados
#     print("\n🎯 RECOMENDAÇÕES DE TRADING:")
#     print("=" * 60)
#     for res in resultados:
#         status = "✅ OPERAR" if res['operar'] else "⏸️ AGUARDAR"
#         direcao = "📈 ALTA" if res['previsao'] == 1 else "📉 BAIXA"
#         print(
#             f"{res['ticker']}: {status} | {direcao} | Prob: {res['probabilidade']:.3f} | Preço: R$ {res['ultimo_preco']:.2f}")

if __name__ == "__main__":
    main()
