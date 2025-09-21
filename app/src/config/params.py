from typing import List, Dict, Any, Tuple


class Params:
    """Configurações globais e hiperparâmetros do sistema de trading."""

    # --- Configurações de Dados ---
    PERIODO_DADOS: str = "3y"
    INTERVALO_DADOS: str = "1d"
    MINIMO_DADOS_TREINO: int = 300

    TICKERS: List[str] = ["PETR4.SA", "VALE3.SA", "ITSA4.SA", "TAEE11.SA", "BBSE3.SA", "BBDC4.SA"]

    # --- Configurações de Feature Engineering e Labeling ---
    # Parâmetros da Tripla Barreira
    TRIPLE_BARRIER_LOOKAHEAD_DAYS: int = 5  # Quantos dias olhar à frente

    # Dicionário de fatores ATR por ticker (fator_lucro, fator_perda)
    ATR_FACTORS: Dict[str, Tuple[float, float]] = {
        "PETR4.SA": (1.6, 1.1),
        "VALE3.SA": (1.5, 1.2),
        "ITSA4.SA": (1.2, 1.0),
        "TAEE11.SA": (1.0, 0.8),
        "BBSE3.SA": (1.5, 1.0),
        "BBDC4.SA": (1.5, 1.2),
        "DEFAULT": (1.5, 1.0)
    }

    # --- Configurações do Modelo ---
    N_FEATURES_A_SELECIONAR: int = 30
    RANDOM_STATE: int = 42

    CONFIDENCE_THRESHOLDS: Dict[str, float] = {
        "PETR4.SA": 0.70,  # Aumentar thresholds
        "VALE3.SA": 0.75,
        "ITSA4.SA": 0.68,
        "TAEE11.SA": 0.70,
        "BBSE3.SA": 0.68,
        "BBDC4.SA": 0.70,
        "DEFAULT": 0.75
    }

    # --- Configurações de Validação e Otimização ---
    N_SPLITS_CV: int = 5  # Splits para validação cruzada temporal
    PURGE_DAYS: int = 5 # Dias para purgar entre treino e teste para evitar data leakage
    EMBARGO_DAYS: int = 3  # Dias de embargo após o teste

    # Parâmetros do Optuna
    OPTUNA_N_TRIALS: int = 50  # Número de tentativas de otimização
    OPTUNA_TIMEOUT_SECONDS: int = 300  # Tempo máximo por otimização

    # --- Configurações de Risco e Backtesting ---
    CUSTO_POR_TRADE_PCT: float = 0.001  # 0.1% de custo por operação (entrada + saída)
    DIAS_UTEIS_ANUAIS: int = 252

    # --- Configurações de Paths ---
    PATH_MODELOS: str = "modelos_treinados"
    PATH_DB_MERCADO: str = "dados/dados_mercado.db"
    PATH_DB_METADATA: str = "dados/model_metadata.db"

    # --- Configurações de Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

    # --- Modelos Base para o Ensemble ---
    LGBM_BASE_PARAMS: Dict[str, Any] = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 400,  # reduzido (antes 500)
        'learning_rate': 0.05,  # mais rápido, menos overfit
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'class_weight': 'balanced',
        'min_child_samples': 100,  # mais alto -> menos overfit
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,  # regularização L1 mais forte
        'reg_lambda': 1.0,  # regularização L2 mais forte
        'max_depth': 5,  # menor profundidade
    }

