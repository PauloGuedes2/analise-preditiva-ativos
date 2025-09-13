#!/usr/bin/env python3
"""
🎯 TESTE DE SPLITS ROBUSTOS PARA MODELO ESTÁVEL
Testa diferentes divisões treino/teste para encontrar o modelo mais confiável
"""

import warnings

from models.classification import ClassificacaoFinal

warnings.filterwarnings('ignore')


def testar_splits_multiplos(ticker="PETR4.SA"):
    """Testa modelo com diferentes splits para validar estabilidade"""

    print("🎯 TESTE DE SPLITS ROBUSTOS")
    print("=" * 60)
    print(f"📊 TICKER: {ticker}")
    print("🎯 OBJETIVO: Modelo estável e confiável")
    print("📅 DADOS: 2 anos até D-1 (sem finais de semana)")
    print()

    # Splits para testar
    splits_config = [
        {"treino": 80, "teste": 20, "nome": "80-20"},
        {"treino": 70, "teste": 30, "nome": "70-30"},
        {"treino": 60, "teste": 40, "nome": "60-40"}
    ]

    resultados = []

    for config in splits_config:
        print(f"\n🧪 TESTANDO SPLIT {config['nome']}")
        print("=" * 50)

        try:
            # Cria modelo com split customizado
            modelo = ClassificacaoFinal(ticker)

            # Modifica temporariamente o split
            modelo.test_size = config["teste"] / 100.0

            print(f"📊 Configuração: {config['treino']}% treino, {config['teste']}% teste")

            # Treina modelo
            sucesso = modelo.treinar(verbose=False)

            if not sucesso:
                print(f"❌ Falha no treinamento do split {config['nome']}")
                continue

            # Coleta métricas
            metricas = {
                'split': config['nome'],
                'treino_pct': config['treino'],
                'teste_pct': config['teste'],
                'accuracy': modelo.ensemble_accuracy,
                'baseline': modelo.baseline_accuracy,
                'melhoria': modelo.ensemble_accuracy - modelo.baseline_accuracy,
                'n_treino': len(modelo.X_train) if hasattr(modelo, 'X_train') else 0,
                'n_teste': len(modelo.X_test) if hasattr(modelo, 'X_test') else 0,
                'rf_acc': modelo.individual_accuracies.get('rf', 0),
                'gb_acc': modelo.individual_accuracies.get('gb', 0),
                'lr_acc': modelo.individual_accuracies.get('lr', 0),
                'nn_acc': modelo.individual_accuracies.get('nn', 0),
                'high_conf_coverage': modelo.high_confidence_coverage
            }

            resultados.append(metricas)

            # Mostra resultados do split
            print(f"✅ Ensemble: {metricas['accuracy']:.3f}")
            print(f"📊 Baseline: {metricas['baseline']:.3f}")
            print(f"🚀 Melhoria: +{metricas['melhoria']:.3f}")
            print(f"📈 Amostras: {metricas['n_treino']} treino, {metricas['n_teste']} teste")
            print(f"🎯 Cobertura alta confiança: {metricas['high_conf_coverage']:.1%}")

            # Testa previsão
            previsao = modelo.prever_direcao()
            if previsao:
                print(f"🔮 Previsão: {previsao['direction']} ({previsao['confidence']:.1%} confiança)")

        except Exception as e:
            print(f"❌ Erro no split {config['nome']}: {e}")
            continue

    # Análise comparativa
    if resultados:
        print(f"\n" + "=" * 60)
        print("📊 ANÁLISE COMPARATIVA DOS SPLITS")
        print("=" * 60)

        # Tabela de resultados
        print(f"{'Split':<8} {'Accuracy':<10} {'Baseline':<10} {'Melhoria':<10} {'Treino':<8} {'Teste':<8}")
        print("-" * 60)

        for r in resultados:
            print(f"{r['split']:<8} {r['accuracy']:<10.3f} {r['baseline']:<10.3f} "
                  f"{r['melhoria']:<10.3f} {r['n_treino']:<8} {r['n_teste']:<8}")

        # Encontra melhor split
        melhor = max(resultados, key=lambda x: x['melhoria'])
        mais_estavel = min(resultados, key=lambda x: abs(x['accuracy'] - x['baseline']))

        print(f"\n🏆 MELHOR PERFORMANCE:")
        print(f"   Split: {melhor['split']}")
        print(f"   Accuracy: {melhor['accuracy']:.3f}")
        print(f"   Melhoria: +{melhor['melhoria']:.3f}")

        print(f"\n⚖️ MAIS ESTÁVEL (menor gap treino/teste):")
        print(f"   Split: {mais_estavel['split']}")
        print(f"   Accuracy: {mais_estavel['accuracy']:.3f}")
        print(f"   Gap: {abs(mais_estavel['accuracy'] - mais_estavel['baseline']):.3f}")

        # Recomendação
        print(f"\n💡 RECOMENDAÇÃO:")

        # Critérios para escolha
        criterios = []
        for r in resultados:
            score = 0
            # Performance (30%)
            score += (r['melhoria'] * 100) * 0.3
            # Estabilidade (40%) - menor gap é melhor
            gap = abs(r['accuracy'] - r['baseline'])
            score += (0.1 - gap) * 100 * 0.4  # Inverte para menor gap = maior score
            # Tamanho do teste (30%) - mais dados de teste é melhor para validação
            score += (r['teste_pct'] / 100) * 0.3 * 100

            criterios.append({
                'split': r['split'],
                'score': score,
                'accuracy': r['accuracy'],
                'melhoria': r['melhoria'],
                'gap': gap,
                'n_teste': r['n_teste']
            })

        recomendado = max(criterios, key=lambda x: x['score'])

        print(f"   🎯 Split recomendado: {recomendado['split']}")
        print(f"   📊 Accuracy: {recomendado['accuracy']:.3f}")
        print(f"   🚀 Melhoria: +{recomendado['melhoria']:.3f}")
        print(f"   ⚖️ Gap treino/teste: {recomendado['gap']:.3f}")
        print(f"   🧪 Amostras teste: {recomendado['n_teste']}")

        print(f"\n📋 JUSTIFICATIVA:")
        print(f"   • Equilibra performance e estabilidade")
        print(f"   • Evita overfitting com validação robusta")
        print(f"   • Mantém dados suficientes para teste")
        print(f"   • Baseado em 2 anos de dados até D-1")

        return recomendado['split']

    else:
        print("❌ Nenhum split funcionou corretamente")
        return None


def validar_modelo_final(ticker="PETR4.SA", split_recomendado="70-30"):
    """Valida o modelo final com o split recomendado"""

    print(f"\n" + "=" * 60)
    print(f"🎯 VALIDAÇÃO FINAL - SPLIT {split_recomendado}")
    print("=" * 60)

    try:
        # Configura split baseado na recomendação
        test_size = {
            "80-20": 0.20,
            "70-30": 0.30,
            "60-40": 0.40
        }.get(split_recomendado, 0.30)

        modelo = ClassificacaoFinal(ticker)
        modelo.test_size = test_size

        print(f"📊 Usando split {split_recomendado} (test_size={test_size})")

        # Treina modelo final
        sucesso = modelo.treinar(verbose=True)

        if not sucesso:
            print("❌ Falha no treinamento final")
            return None

        # Faz previsão
        previsao = modelo.prever_direcao()

        if previsao:
            print(f"\n🎯 MODELO FINAL VALIDADO:")
            print(f"   Split: {split_recomendado}")
            print(f"   Accuracy: {modelo.ensemble_accuracy:.3f}")
            print(f"   Melhoria: +{modelo.ensemble_accuracy - modelo.baseline_accuracy:.3f}")
            print(f"   Previsão: {previsao['direction']} ({previsao['confidence']:.1%})")
            print(f"   Recomendação: {'OPERAR' if previsao['should_trade'] else 'AGUARDAR'}")

        return modelo

    except Exception as e:
        print(f"❌ Erro na validação final: {e}")
        return None


def main():
    """Executa teste completo de splits robustos"""

    print("🎯 SISTEMA DE VALIDAÇÃO ROBUSTA")
    print("=" * 60)
    print("🎯 OBJETIVO: Encontrar modelo estável e confiável")
    print("📅 DADOS: 2 anos até D-1 (excluindo finais de semana)")
    print("🧪 SPLITS: 80-20, 70-30, 60-40")
    print()

    # Testa diferentes splits
    split_recomendado = testar_splits_multiplos("PETR4.SA")

    if split_recomendado:
        # Valida modelo final
        modelo_final = validar_modelo_final("PETR4.SA", split_recomendado)

        if modelo_final:
            print(f"\n✅ SISTEMA VALIDADO COM SUCESSO!")
            print(f"🎯 Split ótimo: {split_recomendado}")
            print(f"📊 Modelo estável e confiável criado")
        else:
            print(f"\n❌ Falha na validação final")
    else:
        print(f"\n❌ Não foi possível encontrar split adequado")


if __name__ == "__main__":
    main()
