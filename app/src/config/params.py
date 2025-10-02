import os
from typing import List, Dict, Tuple


class Params:
    """
    Configurações globais e hiperparâmetros do sistema de trading.

    Esta classe centraliza todos os parâmetros ajustáveis, facilitando a
    manutenção e experimentação sem a necessidade de alterar o código principal.
    """

    # --- Configurações de Dados ---
    PERIODO_DADOS: str = "3y"  # Período para baixar dados (ex: "5y", "1mo")
    INTERVALO_DADOS: str = "1d"  # Intervalo dos candles (ex: "1d", "1h")
    MINIMO_DADOS_TREINO: int = 300  # Número mínimo de registros para treinar um modelo

    # Lista de tickers a serem treinados e avaliados
    TICKERS: List[str] = [
       "ITSA4.SA", "VALE3.SA", "TAEE11.SA"
    ]

    # --- Configurações de Feature Engineering e Labeling ---
    TRIPLE_BARRIER_LOOKAHEAD_DAYS: int = 5  # Janela de tempo (dias) para a metodologia da Tripla Barreira (olhar para frente)

    # Fatores de ATR para definir as barreiras de lucro (take profit) e perda (stop loss).
    ATR_FACTORS: Dict[str, Tuple[float, float]] = {
        "DEFAULT": (1.5, 1.0),  # (Fator Take Profit, Fator Stop Loss)
        "ITSA4.SA": (1.2, 0.8),
        "VALE3.SA": (1.3, 0.8),
        "TAEE11.SA": (1.5, 1.0),
    }

    # --- Configurações do Modelo ---
    N_FEATURES_A_SELECIONAR: int = 18  # Número de features a serem selecionadas pelo modelo
    RANDOM_STATE: int = 42  # Semente para garantir reprodutibilidade

    # --- Configurações de Validação e Otimização ---
    N_SPLITS_CV: int = 5  # Número de folds para a validação cruzada
    PURGE_DAYS: int = 5  # Dias de purga entre treino e teste para evitar data leakage

    # Parâmetros do otimizador de hiperparâmetros Optuna
    OPTUNA_N_TRIALS: int = 100  # Número de tentativas de otimização
    OPTUNA_TIMEOUT_SECONDS: int = 300  # Tempo máximo para a otimização

    # --- Configurações de Risco e Backtesting ---
    CUSTO_POR_TRADE_PCT: float = 0.001  # Custo percentual por operação (ida e volta)
    DIAS_UTEIS_ANUAIS: int = 252  # Usado para anualizar métricas como Sharpe Ratio

    # --- Configurações de Paths ---
    SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PATH_MODELOS: str = os.path.join(SRC_ROOT, "modelos_treinados")  # Diretório para salvar modelos treinados
    PATH_DB_MERCADO: str = os.path.join(SRC_ROOT, "dados", "dados_mercado.db")  # Arquivo do banco de dados

    # --- Configurações de Logging ---
    LOG_LEVEL: str = "INFO"  # Nível de log (DEBUG, INFO, WARNING, ERROR)
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

    # --- Configurações de UI ---
    UI_TAB_NAMES: List[str] = [
        "🎯 **Resumo Executivo**",
        "✅ **Avaliação do Modelo**",
        "🔍 **Análise da Previsão (SHAP)**",
        "🩺 **Saúde do Modelo**",
        "📈 **Análise de Mercado**",
        "📊 **Simulação (In-Sample)**"
    ]

    UI_DRIFT_KEY_FEATURES: List[str] = ['rsi_14', 'vol_21d', 'sma_ratio_10_50', 'momentum_5d']
    UI_VALIDATION_DAYS: int = 30  # Número de dias para validação recente na UI
