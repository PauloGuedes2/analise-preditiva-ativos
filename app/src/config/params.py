from typing import List, Dict, Any


class Params:
    """Configurações globais do sistema de trading."""

    # Configurações de dados
    PERIODO_DADOS: str = "3y"
    INTERVALO_DADOS: str = "1d"
    MINIMO_DADOS_TREINO: int = 100

    # Tickers disponíveis
    TICKERS: List[str] = [
        "PETR4.SA", "VALE3.SA", "ITSA4.SA",
        "TAEE11.SA", "BBSE3.SA", "ABEV3.SA"
    ]

    # Configurações do modelo
    N_FEATURES: int = 25
    RANDOM_STATE: int = 42
    CONFIDENCE_OPERAR: float = 0.60
    OTIMIZAR_HIPERPARAMETROS: bool = True

    # Validação walk-forward
    N_SPLITS: int = 4
    PURGE_DAYS: int = 2

    # Configurações de risco
    CUSTO_POR_TRADE_PCT: float = 0.0005
    NIVEL_CONFIANCA_VAR: float = 0.95

    # Configurações de features
    INDICADORES_CONFIG: Dict[str, int] = {
        'rsi_periodo': 14,
        'stoch_periodo': 14,
        'bollinger_periodo': 20,
        'atr_periodo': 14,
        'cmf_periodo': 20
    }

    # Janelas para cálculos técnicos
    JANELAS_MEDIAS_MOVEIS: List[int] = [5, 10, 20, 50]
    JANELAS_RETORNOS: List[int] = [1, 3, 5]
    JANELAS_VOLATILIDADE: List[int] = [5, 20]
    JANELA_CORRELACAO: int = 20

    # Paths de arquivos
    PATH_MODELOS: str = "modelos_treinados"
    PATH_DB_MERCADO: str = "dados_mercado.db"
    PATH_DB_METADATA: str = "model_metadata.db"

    # Configurações de logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

    # Configurações de visualização
    CORES_GRAFICOS: Dict[str, str] = {
        'preco': '#1f77b4',
        'sinal_operacao': 'limegreen',
        'equidade': '#2ca02c',
        'sinal_contorno': 'darkgreen'
    }

    TAMANHOS_MARCACORES: Dict[str, int] = {
        'sinal_operacao': 10,
        'linha_preco': 2,
        'linha_equidade': 3
    }

    DIAS_UTEIS_ANUAIS: int = 252


class ModelParams:
    """Parâmetros específicos para cada tipo de modelo."""

    RANDOM_FOREST: Dict[str, Any] = {
        'n_estimators': 300,
        'n_jobs': -1,
        'random_state': Params.RANDOM_STATE
    }

    GRADIENT_BOOSTING: Dict[str, Any] = {
        'n_estimators': 200,
        'random_state': Params.RANDOM_STATE
    }

    LOGISTIC_REGRESSION: Dict[str, Any] = {
        'max_iter': 1000,
        'random_state': Params.RANDOM_STATE
    }

    MLP: Dict[str, Any] = {
        'hidden_layer_sizes': (30, 15),
        'alpha': 0.1,
        'max_iter': 1000,
        'random_state': Params.RANDOM_STATE,
        'early_stopping': True
    }

    # Grades para otimização de hiperparâmetros
    GRADES_OTIMIZACAO: Dict[str, Dict[str, Any]] = {
        'rf': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'gb': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        },
        'lr': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        },
        'nn': {
            'hidden_layer_sizes': [(50,), (100,), (50, 30)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
    }