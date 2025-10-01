# 📈 Tech Challenge 3: Machine Learning para Trading Quantitativo

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Ativo-brightgreen)]()

---

## 📚 Índice

- [Descrição](#descrição)
- [Funcionalidades Principais](#funcionalidades-principais)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação e Requisitos](#instalação-e-requisitos)
- [Como Rodar](#como-rodar)
- [Configuração (`params.py`)](#configuração-paramspy)
- [Pipeline de ML](#pipeline-de-ml)
- [Critérios de Salvamento do Modelo](#critérios-de-salvamento-do-modelo)
- [Saídas e Resultados](#saídas-e-resultados)
- [Uso da UI (Streamlit)](#uso-da-ui-streamlit)
- [Exemplos Práticos](#exemplos-práticos)
- [Métricas e Interpretação](#métricas-e-interpretação)
- [Troubleshooting (Erros Comuns)](#troubleshooting-erros-comuns)
- [Melhorias Futuras / Roadmap](#melhorias-futuras--roadmap)
- [Licença](#licença)
- [Aviso Legal](#aviso-legal)

---

## 📝 Descrição

Este projeto é uma plataforma open-source para desenvolvimento, treinamento e avaliação de modelos de Machine Learning aplicados ao trading quantitativo de ativos financeiros brasileiros. O objetivo é fornecer um ambiente robusto e modular para pesquisa, aprendizado e prototipagem de estratégias quantitativas, com ênfase em boas práticas de validação, backtesting e análise de risco.

Destinado a desenvolvedores quantitativos, estudantes de Data Science, e traders que desejam explorar técnicas modernas de ML, o projeto integra coleta automática de dados, engenharia de features, labeling avançado (Tripla Barreira), otimização de modelos com Optuna, validação temporal rigorosa e uma interface interativa via Streamlit.

Diferenciais incluem: pipeline completo e automatizado, validação walk-forward com purging, backtesting vetorial, seleção automática de features, explicabilidade via SHAP, e atualização contínua dos dados. Tudo isso com logging detalhado, persistência eficiente e documentação clara.

---

## 🚀 Funcionalidades Principais

- 📊 Coleta automática de dados do Yahoo Finance com cache em SQLite
- 🛠️ Engenharia de features técnicas e de mercado (RSI, ATR, OBV, SMA, volatilidade, correlação, IBOV)
- 🏷️ Labeling com Tripla Barreira baseada em ATR
- 🤖 Treinamento de modelos LightGBM multiclasse com Optuna (Sharpe Ratio)
- 🧠 Seleção automática de features (SelectFromModel)
- 🔬 Validação robusta via Purged Walk-Forward CV
- 📈 Backtesting vetorial com métricas de risco (Sharpe, Sortino, Max Drawdown, etc.)
- 🎯 Calibração de threshold operacional
- 💾 Persistência de modelos em `modelos_treinados/`
- 🖥️ Interface Streamlit com múltiplas abas analíticas
- 🔄 Serviço de atualização automática de dados em thread
- 📝 Logging customizado com níveis, formatos e data

---

## 🗂️ Estrutura do Projeto

```text
tech3/
├── app/
│   ├── run.py                # Inicia a aplicação Streamlit
│   ├── train.py              # Executa pipeline de treinamento
│   ├── txt.py                # Exporta código para project_code.txt
│   ├── requirements.txt      # Dependências do projeto
│   ├── project_code.txt      # Código exportado
│   ├── dados/
│   │   └── dados_mercado.db  # Banco de dados SQLite (OHLCV)
│   ├── modelos_treinados/
│   │   └── *.joblib          # Modelos treinados
│   └── src/
│       ├── config/
│       │   └── params.py     # Configurações globais
│       ├── data/
│       │   ├── data_loader.py    # Coleta e cache de dados
│       │   └── data_updater.py   # Atualização automática (thread)
│       ├── models/
│       │   ├── classification.py     # ClassificadorTrading (LightGBM)
│       │   ├── feature_engineer.py   # Engenharia de features e labels
│       │   ├── technical_indicators.py # Indicadores técnicos
│       │   └── validation.py         # Purged K-Fold CV
│       ├── backtesting/
│       │   └── risk_analyzer.py      # Backtesting vetorial e métricas
│       ├── training/
│       │   └── training_pipeline.py  # Pipeline de treino (WFV)
│       └── ui/
│           ├── app.py                # Interface Streamlit
│           └── dashboard_view.py     # Visualizações da UI
```

| Diretório/Arquivo                    | Função Principal                                  |
|--------------------------------------|---------------------------------------------------|
| `run.py`                             | Inicia a interface Streamlit                      |
| `train.py`                           | Executa pipeline de treinamento                   |
| `src/config/params.py`               | Configurações globais                             |
| `src/data/data_loader.py`            | Coleta e cache de dados financeiros               |
| `src/data/data_updater.py`           | Atualização automática dos dados                  |
| `src/models/classification.py`       | ClassificadorTrading com LightGBM                 |
| `src/models/feature_engineer.py`     | Engenharia de features e labeling                 |
| `src/models/technical_indicators.py` | Indicadores técnicos                              |
| `src/models/validation.py`           | Validação temporal (Purged K-Fold CV)             |
| `src/backtesting/risk_analyzer.py`   | Backtesting e métricas de risco                   |
| `src/training/training_pipeline.py`  | Pipeline de treino e validação                    |
| `src/ui/app.py`                      | Interface Streamlit                               |
| `src/ui/dashboard_view.py`           | Visualizações analíticas                          |
| `dados/dados_mercado.db`             | Banco de dados SQLite (OHLCV)                     |
| `modelos_treinados/*.joblib`         | Modelos treinados e persistidos                   |

---

## ⚙️ Instalação e Requisitos

- **Python**: 3.10 ou superior
- **Dependências**:  
  - `streamlit`
  - `yfinance`
  - `lightgbm`
  - `optuna`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `joblib`
  - `matplotlib`
  - `plotly`
  - `shap`

Instale as dependências com:

```bash
pip install -r app/requirements.txt
```

---

## ▶️ Como Rodar

### 1. Iniciar a Interface Streamlit

```bash
python app/run.py
```

- Abre a UI interativa no navegador.
- Logs detalhados no console.

### 2. Treinar Modelos

```bash
python app/train.py
```

- Executa pipeline completo de treinamento.
- Modelos salvos em `modelos_treinados/`.

#### Exemplos de Saída

- Modelos: `modelos_treinados/modelo_ITSA4.SA.joblib`
- Dados: `dados/dados_mercado.db`
- Logs: exibidos no console

---

## 🛠️ Configuração (`params.py`)

Principais parâmetros globais:

| Parâmetro         | Valor Default         | Descrição                       |
|-------------------|-----------------------|---------------------------------|
| `tickers`         | VALE3, ITSA4, ELET3   | Ativos analisados               |
| `periodo`         | 3y                    | Período histórico (3 anos)      |
| `intervalo`       | 1d                    | Intervalo dos dados (diário)    |
| `custo_trade`     | 0.1%                  | Custo por operação              |
| `optuna_trials`   | 100                   | Nº de tentativas Optuna         |
| `optuna_timeout`  | 300s                  | Tempo máximo por otimização     |
| `model_params`    | ...                   | Hiperparâmetros do LightGBM     |
| `ui_params`       | ...                   | Configurações da interface      |
| `logging`         | customizado           | Níveis, formatos e data         |

---

## 🔄 Pipeline de ML

1. **Coleta de Dados**  
   - Yahoo Finance via `yfinance`
   - Cache em SQLite (`dados_mercado.db`)
2. **Engenharia de Features**  
   - Indicadores técnicos (RSI, ATR, OBV, Bollinger, SMA, volatilidade, correlação, IBOV)
3. **Labeling Tripla Barreira**  
   - Geração de labels multiclasse baseada em ATR
4. **Treinamento de Modelos**  
   - LightGBM + Optuna (Sharpe Ratio)
   - Seleção automática de features
5. **Validação Temporal**  
   - Purged Walk-Forward CV
6. **Backtesting Vetorial**  
   - Métricas: Sharpe, Sortino, Win Rate, Max Drawdown, Payoff, Profit Factor
7. **Persistência de Modelos**  
   - Salvamento em `modelos_treinados/`
8. **Interface Interativa**  
   - Streamlit com múltiplas abas analíticas

---

## 🏆 Critérios de Salvamento do Modelo

- **F1 Score** > 0.50
- **Sharpe Ratio** > -0.1
- **Trades** ≥ 2.5

Modelos que não atendem aos critérios são descartados.

---

## 📦 Saídas e Resultados

| Saída                      | Localização                      | Descrição                       |
|----------------------------|----------------------------------|---------------------------------|
| Modelos treinados          | `modelos_treinados/*.joblib`     | Modelos LightGBM otimizados     |
| Dados de mercado           | `dados/dados_mercado.db`         | OHLCV e features                |
| Logs                       | Console                          | Execução, erros, métricas       |

---

## 🖥️ Uso da UI (Streamlit)

A interface possui as seguintes abas:

- **Resumo Executivo**: visão geral dos resultados e métricas
- **Avaliação do Modelo**: performance, confusion matrix, ROC, etc.
- **SHAP**: explicabilidade dos modelos
- **Saúde do Modelo**: monitoramento de overfitting, drift, etc.
- **Análise de Mercado**: gráficos, correlações, volatilidade
- **Simulação In-Sample**: backtest e simulação de estratégias

---

## 💡 Exemplos Práticos

### 1. Rodar Backtest Manual

```python
from src.backtesting.risk_analyzer import RiskAnalyzer
ra = RiskAnalyzer(...)
ra.run_backtest()
```

### 2. Prever Direção

```python
from src.models.classification import ClassificadorTrading
clf = ClassificadorTrading(...)
direcao = clf.prever_direcao(dados)
```

### 3. Atualizar Dados

```python
from src.data.data_updater import DataUpdater
updater = DataUpdater(...)
updater.start()
```

---

## 📊 Métricas e Interpretação

| Métrica         | Descrição                                      |
|-----------------|------------------------------------------------|
| Sharpe Ratio    | Retorno ajustado ao risco                      |
| Sortino Ratio   | Retorno ajustado ao risco de perdas            |
| Max Drawdown    | Máxima perda acumulada                         |
| Profit Factor   | Razão entre ganhos e perdas                    |
| Payoff          | Ganho médio por trade                          |
| Win Rate        | % de trades vencedores                         |

---

## 🛠️ Troubleshooting (Erros Comuns)

- **Timeout no yfinance**: verifique conexão ou aumente timeout
- **Dataset vazio**: cheque tickers, período e intervalo
- **Erro no PurgedKFoldCV**: dados insuficientes ou formato incorreto
- **Problema de conexão SQLite**: arquivo bloqueado ou permissões
- **Falta de dados suficientes**: ajuste período ou tickers

---

## 🚧 Melhorias Futuras / Roadmap

- Integração com corretoras (API)
- Suporte multi-ativos e portfólios
- Deploy em nuvem (Streamlit Cloud, Heroku)
- Novas métricas de risco e performance
- Novos modelos (XGBoost, redes neurais)
- Monitoramento em tempo real

---

## 📄 Licença

Distribuído sob a Licença MIT. Veja [LICENSE](LICENSE) para mais detalhes.

---

## ⚠️ Aviso Legal

Este projeto é para fins educacionais e de pesquisa. Não constitui recomendação de investimento ou consultoria financeira. Use por sua conta e risco.

---

