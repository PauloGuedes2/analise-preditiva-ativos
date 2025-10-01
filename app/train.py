from src.logger.logger import logger
from src.training.training_pipeline import TreinadorModelos


def main() -> None:
    """Função principal que instancia e executa a pipeline de treinamento."""
    try:
        treinador = TreinadorModelos()
        treinador.executar_treinamento_completo()
    except Exception as e:
        logger.exception(f"Ocorreu um erro fatal no processo de treinamento: {e}")


if __name__ == "__main__":
    main()
