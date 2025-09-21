from typing import List, Dict, Tuple


class Params:
    """Configurações globais e hiperparâmetros do sistema de trading."""

    # --- Configurações de Dados ---
    PERIODO_DADOS: str = "3y"
    INTERVALO_DADOS: str = "1d"
    MINIMO_DADOS_TREINO: int = 300

    TICKERS: List[str] = [
        "PETR4.SA", "VALE3.SA", "ITSA4.SA", "ELET3.SA", "ITUB4.SA"
    ]

    # --- Configurações de Feature Engineering e Labeling ---
    # Parâmetros da Tripla Barreira
    TRIPLE_BARRIER_LOOKAHEAD_DAYS: int = 5  # Quantos dias olhar à frente

    # É possível ajustar os fatores de ATR para cada ativo individualmente. O padrão é (1.5, 1.0) para todos os ativos.
    ATR_FACTORS: Dict[str, Tuple[float, float]] = {
        "DEFAULT": (1.5, 1.0),
        "PETR4.SA": (1.5, 1.0),
        "ITSA4.SA": (1.5, 1.0),
        "ITUB4.SA": (1.4, 1.0),
        "ELET3.SA": (1.2, 0.8),
        "VALE3.SA": (1.2, 0.8)
    }

    # --- Configurações do Modelo ---
    N_FEATURES_A_SELECIONAR: int = 30
    RANDOM_STATE: int = 42

    # --- Configurações de Validação e Otimização ---
    N_SPLITS_CV: int = 5
    PURGE_DAYS: int = 5
    EMBARGO_DAYS: int = 3

    # Parâmetros do Optuna
    OPTUNA_N_TRIALS: int = 100
    OPTUNA_TIMEOUT_SECONDS: int = 300

    # --- Configurações de Risco e Backtesting ---
    CUSTO_POR_TRADE_PCT: float = 0.001
    DIAS_UTEIS_ANUAIS: int = 252

    # --- Configurações de Paths ---
    PATH_MODELOS: str = "modelos_treinados"
    PATH_DB_MERCADO: str = "dados/dados_mercado.db"
    PATH_DB_METADATA: str = "dados/model_metadata.db"

    # --- Configurações de Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
