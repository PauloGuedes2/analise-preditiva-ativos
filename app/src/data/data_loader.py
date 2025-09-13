from datetime import datetime, timedelta

import yfinance as yf

from src.data.database_manager import GerenciadorBancoDados


class CarregadorDados:
    """
    Classe responsável por carregar dados de ações do Yahoo Finance e gerenciar cache local.
    
    Esta classe fornece métodos para baixar, armazenar e recuperar dados históricos de ações,
    com sistema de fallback para garantir disponibilidade dos dados.
    """
    
    def __init__(self):
        """
        Inicializa o carregador de dados.
        
        Cria uma instância do gerenciador de banco de dados para cache local.
        """
        self.gerenciador_bd = GerenciadorBancoDados()

    def baixar_dados(self, ticker, data_inicio, data_fim):
        """
        Baixa dados históricos do Yahoo Finance e salva no banco de dados local.
        
        Args:
            ticker (str): Símbolo da ação (ex: 'PETR4.SA')
            data_inicio (str): Data de início no formato 'YYYY-MM-DD'
            data_fim (str): Data de fim no formato 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: DataFrame com dados históricos da ação
        """
        print(f"Baixando dados para {ticker} de {data_inicio} a {data_fim}")

        dt_inicio = datetime.strptime(data_inicio, '%Y-%m-%d')
        dt_fim = datetime.strptime(data_fim, '%Y-%m-%d')

        dados = yf.download(ticker, start=dt_inicio, end=dt_fim, progress=False)

        if dados.empty:
            print(f"Nenhum dado encontrado para {ticker} no período especificado")
            print("Usando dados históricos para teste...")
            data_fim_teste = datetime.now().strftime('%Y-%m-%d')
            data_inicio_teste = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            dados = yf.download(ticker, start=data_inicio_teste, end=data_fim_teste, progress=False)

        if not dados.empty:
            self.gerenciador_bd.salvar_dados_acao(dados, ticker)

        return dados

    def obter_dados(self, ticker, data_inicio, data_fim):
        """
        Obtém dados históricos, primeiro tentando do cache local, depois baixando se necessário.
        
        Args:
            ticker (str): Símbolo da ação (ex: 'PETR4.SA')
            data_inicio (str): Data de início no formato 'YYYY-MM-DD'
            data_fim (str): Data de fim no formato 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: DataFrame com dados históricos da ação
        """
        dados_bd = self.gerenciador_bd.carregar_dados_acao(ticker)

        if not dados_bd.empty:
            dt_inicio = datetime.strptime(data_inicio, '%Y-%m-%d')
            dt_fim = datetime.strptime(data_fim, '%Y-%m-%d')

            dados_bd = dados_bd[(dados_bd.index >= dt_inicio) & (dados_bd.index <= dt_fim)]

            if not dados_bd.empty:
                print(f"Dados carregados do banco: {len(dados_bd)} registros")
                return dados_bd

        return self.baixar_dados(ticker, data_inicio, data_fim)

    def obter_dados_com_fallback(self, ticker, data_inicio, data_fim):
        """
        Obtém dados com sistema de fallback para período mais recente se necessário.
        
        Se não encontrar dados no período solicitado, tenta buscar dados do último ano.
        
        Args:
            ticker (str): Símbolo da ação (ex: 'PETR4.SA')
            data_inicio (str): Data de início no formato 'YYYY-MM-DD'
            data_fim (str): Data de fim no formato 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: DataFrame com dados históricos da ação
        """
        try:
            dados = self.obter_dados(ticker, data_inicio, data_fim)

            if dados.empty:
                print(f"Nenhum dado encontrado para {ticker} no período {data_inicio} a {data_fim}")
                print("Buscando dados do último ano...")

                data_fim_fallback = datetime.now().strftime('%Y-%m-%d')
                data_inicio_fallback = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

                dados = self.obter_dados(ticker, data_inicio_fallback, data_fim_fallback)

            return dados

        except Exception as e:
            print(f"Erro ao obter dados: {e}")
            return self.baixar_dados(ticker, data_inicio, data_fim)
