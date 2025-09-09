from datetime import datetime, timedelta

import yfinance as yf

from src.data.database_manager import DatabaseManager


class DataLoader:
    def __init__(self):
        self.db_manager = DatabaseManager()

    def download_data(self, ticker, start_date, end_date):
        """Baixa dados do yfinance e salva no banco"""
        print(f"Baixando dados para {ticker} de {start_date} a {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        data = yf.download(ticker, start=start_dt, end=end_dt, progress=False)

        if data.empty:
            print(f"Nenhum dado encontrado para {ticker} no período especificado")
            print("Usando dados históricos para teste...")
            test_end_date = datetime.now().strftime('%Y-%m-%d')
            test_start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            data = yf.download(ticker, start=test_start_date, end=test_end_date, progress=False)

        if not data.empty:
            self.db_manager.save_stock_data(data, ticker)

        return data

    def get_data(self, ticker, start_date, end_date):
        """Obtém dados, baixando se necessário"""
        db_data = self.db_manager.load_stock_data(ticker)

        if not db_data.empty:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            db_data = db_data[(db_data.index >= start_dt) & (db_data.index <= end_dt)]

            if not db_data.empty:
                print(f"Dados carregados do banco: {len(db_data)} registros")
                return db_data

        return self.download_data(ticker, start_date, end_date)

    def get_data_with_fallback(self, ticker, start_date, end_date):
        """Obtém dados com fallback para período mais recente se necessário"""
        try:
            data = self.get_data(ticker, start_date, end_date)

            if data.empty:
                print(f"Nenhum dado encontrado para {ticker} no período {start_date} a {end_date}")
                print("Buscando dados do último ano...")

                fallback_end = datetime.now().strftime('%Y-%m-%d')
                fallback_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

                data = self.get_data(ticker, fallback_start, fallback_end)

            return data

        except Exception as e:
            print(f"Erro ao obter dados: {e}")
            return self.download_data(ticker, start_date, end_date)
