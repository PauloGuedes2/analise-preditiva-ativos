import numpy as np
from sklearn.metrics import accuracy_score


class AnalisadorRisco:
    """
    Classe responsável por calcular métricas de risco e gerar sinais de trading.
    
    Esta classe fornece métodos estáticos para avaliar a performance de estratégias
    de trading e gerar recomendações baseadas nas previsões do modelo.
    """
    
    @staticmethod
    def calcular_metricas_risco(y_teste, y_pred, retornos_teste):
        """
        Calcula métricas abrangentes de risco e performance de trading.
        
        Args:
            y_teste (array-like): Valores reais da direção do preço
            y_pred (array-like): Previsões da direção do preço
            retornos_teste (array-like): Retornos reais observados
            
        Returns:
            dict: Dicionário com métricas de performance calculadas
        """
        resultados = {}

        # Acurácia das previsões
        resultados['acuracia'] = accuracy_score(y_teste, y_pred)
        
        # Lucro total das operações previstas como alta
        resultados['lucro_total'] = retornos_teste[y_pred == 1].sum()

        # Sharpe Ratio (retorno ajustado ao risco)
        retornos_excesso = retornos_teste - 0.0001  # Taxa livre de risco
        resultados['sharpe_ratio'] = retornos_excesso.mean() / retornos_excesso.std() * np.sqrt(252)

        # Drawdown máximo (maior perda acumulada)
        retornos_acumulados = (1 + retornos_teste).cumprod()
        pico = retornos_acumulados.expanding().max()
        drawdown = (retornos_acumulados - pico) / pico
        resultados['drawdown_maximo'] = drawdown.min()

        # Taxa de acerto das operações
        operacoes_vencedoras = retornos_teste[y_pred == 1] > 0
        resultados['taxa_acerto'] = operacoes_vencedoras.mean() if len(operacoes_vencedoras) > 0 else 0

        # Fator de lucro (lucro bruto / perda bruta)
        lucro_bruto = retornos_teste[(y_pred == 1) & (retornos_teste > 0)].sum()
        perda_bruta = abs(retornos_teste[(y_pred == 1) & (retornos_teste < 0)].sum())
        resultados['fator_lucro'] = lucro_bruto / perda_bruta if perda_bruta > 0 else float('inf')

        return resultados

    @staticmethod
    def gerar_sinais_trading(previsao):
        """
        Gera sinais de trading baseados na previsão do modelo.
        
        Args:
            previsao (dict): Dicionário com previsão contendo direção, confiança e retorno esperado
            
        Returns:
            list: Lista de strings com sinais e recomendações de trading
        """
        sinais = []

        # Sinal de direção
        if previsao['direction'] == 'ALTA':
            sinais.append("📈 SINAL: COMPRA")
        else:
            sinais.append("📉 SINAL: VENDA")

        # Força do sinal baseada na confiança
        confianca = previsao['direction_confidence']
        if confianca > 0.7:
            sinais.append("💪 FORTE (Confiança > 70%)")
        elif confianca > 0.6:
            sinais.append("👍 MÉDIO (Confiança 60-70%)")
        else:
            sinais.append("⚠️  FRACO (Confiança < 60%)")

        # Potencial de retorno
        retorno_esperado = previsao['expected_return']
        if retorno_esperado > 0.015:
            sinais.append("🎯 ALTO POTENCIAL (Retorno > 1.5%)")
        elif retorno_esperado > 0.005:
            sinais.append("✅ OPERAR (Retorno 0.5-1.5%)")
        elif retorno_esperado > -0.005:
            sinais.append("⏸️  NEUTRO (Retorno -0.5% a 0.5%)")
        else:
            sinais.append("🚫 EVITAR (Retorno < -0.5%)")

        # Recomendação de tamanho da posição
        if confianca > 0.65 and abs(retorno_esperado) > 0.008:
            tamanho_posicao = "Tamanho: NORMAL"
        elif confianca > 0.75 and abs(retorno_esperado) > 0.015:
            tamanho_posicao = "Tamanho: MAIOR"
        else:
            tamanho_posicao = "Tamanho: REDUZIDO"
        sinais.append(tamanho_posicao)

        return sinais
