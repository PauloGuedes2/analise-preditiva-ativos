"""
🎯 SISTEMA DE CLASSIFICAÇÃO DE DIREÇÃO DE AÇÕES
Modelo focado EXCLUSIVAMENTE em prever direção (Alta/Baixa) para o próximo dia
"""

import warnings

from src.models.classification import ClassificacaoFinal

warnings.filterwarnings('ignore')


def main():
    """Aplicação principal do sistema de classificação"""

    print("🎯 SISTEMA DE CLASSIFICAÇÃO DE DIREÇÃO DE AÇÕES")
    print("=" * 60)
    print("🎯 FOCO: Prever APENAS direção (Alta/Baixa)")
    print("📅 DADOS: Até D-1 com 2 anos de histórico")
    print("🏆 TICKER PRINCIPAL: PETR4.SA")
    print()

    # Ticker principal (melhor performer validado)
    ticker = "PETR4.SA"

    try:
        print(f"🚀 Iniciando sistema para {ticker}...")

        # Cria modelo de classificação
        modelo = ClassificacaoFinal(ticker)

        # Treina modelo
        print(f"\n🤖 TREINANDO MODELO DE CLASSIFICAÇÃO...")
        sucesso = modelo.treinar()

        if not sucesso:
            print("❌ Falha no treinamento!")
            return

        # Faz previsão para amanhã
        print(f"\n🔮 GERANDO PREVISÃO PARA AMANHÃ...")
        previsao = modelo.prever_direcao()

        if previsao is None:
            print("❌ Falha na previsão!")
            return

        # Mostra resultado final
        print(f"\n" + "=" * 60)
        print("🎯 RESULTADO FINAL")
        print("=" * 60)

        print(f"📊 TICKER: {previsao['ticker']}")
        print(f"📅 DADOS ATÉ: {previsao['data_ultima']}")
        print(f"💰 ÚLTIMO PREÇO: ${previsao['last_price']:.2f}")
        print(f"📅 PREVISÃO PARA: {previsao['data_previsao']} ({previsao.get('dia_previsao', 'Próximo dia útil')})")
        print()

        # Sinal principal
        emoji_direcao = "📈" if previsao['direction'] == 'ALTA' else "📉"
        emoji_operacao = "✅" if previsao['should_trade'] else "⚠️"

        print(f"🎯 DIREÇÃO PREVISTA: {emoji_direcao} {previsao['direction']}")
        print(f"📊 PROBABILIDADE: {previsao['probability']:.1%}")
        print(f"🎯 CONFIANÇA: {previsao['confidence']:.1%}")
        print(f"💡 RECOMENDAÇÃO: {emoji_operacao} {'OPERAR' if previsao['should_trade'] else 'AGUARDAR'}")

        if previsao['should_trade']:
            print(f"\n✅ SINAL DE OPERAÇÃO CONFIRMADO!")
            print(f"   🎯 Direção: {previsao['direction']}")
            print(f"   📊 Confiança: {previsao['confidence']:.1%} (≥75% necessário)")
            print(f"   💰 Preço referência: ${previsao['last_price']:.2f}")
            print(f"   ⏰ Executar na abertura do pregão")
        else:
            print(f"\n⚠️ CONFIANÇA INSUFICIENTE - AGUARDAR")
            print(f"   🎯 Confiança atual: {previsao['confidence']:.1%}")
            print(f"   📊 Mínimo necessário: 75%")
            print(f"   💡 Aguardar sinal de maior confiança")

        # Detalhes técnicos
        print(f"\n🤖 DETALHES POR MODELO:")
        for model_name, proba in previsao['individual_probas'].items():
            direction = 'ALTA' if proba > 0.5 else 'BAIXA'
            emoji = "📈" if proba > 0.5 else "📉"
            print(f"   {model_name.upper()}: {emoji} {direction} ({proba:.3f})")

        # Próximos passos
        print(f"\n📋 PRÓXIMOS PASSOS:")
        if previsao['should_trade']:
            print(f"   1. ✅ Executar operação na abertura")
            print(f"   2. 📊 Monitorar resultado durante o dia")
            print(f"   3. 📈 Avaliar performance no fechamento")
            print(f"   4. 🔄 Retreinar modelo com novos dados")
        else:
            print(f"   1. ⏳ Aguardar próximo sinal")
            print(f"   2. 📊 Monitorar evolução da confiança")
            print(f"   3. 🔄 Retreinar modelo amanhã")
            print(f"   4. 🧪 Considerar outros tickers se necessário")

        print(f"\n🎯 SISTEMA EXECUTADO COM SUCESSO!")
        print(f"Status: {'OPERACIONAL' if sucesso else 'ERRO'}")

    except Exception as e:
        print(f"❌ Erro no sistema: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
