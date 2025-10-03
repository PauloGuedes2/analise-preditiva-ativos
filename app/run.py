import os
import sys
import subprocess

class Run:
    """Classe responsável por rodar a aplicação Streamlit."""

    def __init__(self):
        """Inicializa a classe, definindo os caminhos base do projeto."""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.app_path = os.path.join(self.base_dir, "app.py")

    def _verificar_app(self) -> None:
        """Verifica se o arquivo app.py existe, encerrando se não encontrar."""
        if not os.path.isfile(self.app_path):
            print(f"Erro: Arquivo da aplicação não encontrado em: {self.app_path}")
            sys.exit(1)

    def start(self) -> None:
        """Executa a aplicação via Streamlit."""
        self._verificar_app()
        print(f"Iniciando aplicação Streamlit de {self.app_path}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", self.app_path],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar o Streamlit: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"Erro: O comando '{sys.executable}' não foi encontrado.")
            sys.exit(1)


if __name__ == "__main__":
    runner = Run()
    runner.start()