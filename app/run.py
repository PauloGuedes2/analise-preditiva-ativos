import os
import sys
import subprocess


class Run:
    """Classe responsável por rodar a aplicação Streamlit."""

    def __init__(self):
        # Caminho absoluto do app.py
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.app_path = os.path.join(self.base_dir, "src", "ui", "app.py")

    def verificar_app(self):
        """Verifica se app.py existe."""
        if not os.path.isfile(self.app_path):
            print(f"Arquivo não encontrado: {self.app_path}")
            sys.exit(1)

    def start(self):
        """Executa a aplicação via Streamlit."""
        self.verificar_app()
        subprocess.run([sys.executable, "-m", "streamlit", "run", self.app_path])


if __name__ == "__main__":
    runner = Run()
    runner.start()
