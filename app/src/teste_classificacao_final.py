"""
🎯 TESTE DO MODELO DE CLASSIFICAÇÃO FINAL
Testa o modelo definitivo focado EXCLUSIVAMENTE em classificação
"""

from models.classificacao_final import ClassificacaoFinal


def main():
    """Testa o modelo de classificação final"""

    print("🎯 TESTE MODELO DE CLASSIFICAÇÃO FINAL")
    print("=" * 60)
    print("🎯 OBJETIVO: Prever APENAS direção (Alta/Baixa)")
    print("📅 DADOS: Até D-1 com 2 anos de histórico")
    print("🎯 FOCO: PETR4.SA (melhor performer)")
    print()

    # Testa com PETR4.SA (nosso melhor ticker)
    ticker = 'PETR4.SA'

    # Cria modelo
    modelo = ClassificacaoFinal(ticker)

    # Treina modelo
    sucesso = modelo.treinar()

    if sucesso:
        print(f"\n🔮 FAZENDO PREVISÃO PARA AMANHÃ...")

        # Faz previsão
        previsao = modelo.prever_direcao()

        if previsao:
            print(f"\n🎯 RESUMO DA PREVISÃO:")
            print(f"   Ticker: {previsao['ticker']}")
            print(f"   Direção: {previsao['direction']}")
            print(f"   Confiança: {previsao['confidence']:.1%}")
            print(f"   Recomendação: {'OPERAR' if previsao['should_trade'] else 'AGUARDAR'}")

            if previsao['should_trade']:
                print(f"\n✅ SINAL DE OPERAÇÃO DETECTADO!")
                print(f"   📊 Probabilidade: {previsao['probability']:.1%}")
                print(f"   🎯 Confiança: {previsao['confidence']:.1%}")
                print(f"   💰 Preço atual: ${previsao['last_price']:.2f}")
                print(f"   📅 Para: {previsao['data_previsao']}")
            else:
                print(f"\n⚠️ CONFIANÇA INSUFICIENTE - AGUARDAR")
                print(f"   🎯 Confiança atual: {previsao['confidence']:.1%}")
                print(f"   📊 Mínimo necessário: 75%")

        print(f"\n🎯 MODELO DE CLASSIFICAÇÃO TESTADO COM SUCESSO!")

    else:
        print(f"\n❌ FALHA NO TREINAMENTO DO MODELO")


if __name__ == "__main__":
    main()
