# 🚀 **Análise Preditiva de Ativos** 

---
<div align="center">

<!-- Tecnologias Principais -->
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

<!-- Machine Learning -->
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-4285F4?style=for-the-badge&logo=optuna&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge&logoColor=white)

<!-- Interface e Deploy -->
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

<!-- Licença e Status -->
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-🟢_Online-brightgreen?style=for-the-badge)
![Maintained](https://img.shields.io/badge/Maintained-✅_Active-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0-purple?style=for-the-badge)

</div>

---

## 📋 **Índice**

### 🎯 **Visão Geral**
- [O Que Este Sistema Faz](#-o-que-este-sistema-faz)
- [A História Por Trás do Projeto](#-a-história-por-trás-do-projeto)

### 🚀 **Guia de Uso Prático**
- [Instalação Rápida](#-instalação-rápida)
- [Como Usar](#-como-usar)
- [Personalização e Configuração](#-personalização-e-configuração)

### 📊 **Análise e Aplicação**
- [Entendendo os Resultados](#-entendendo-os-resultados)
- [Casos de Uso Reais](#-casos-de-uso-reais)

### 🧠 **Como o Sistema Funciona (Deep Dive)**
- [Como Funciona Por Dentro](#-como-funciona-por-dentro)
- [Arquitetura Técnica](#-arquitetura-técnica)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

### 🛠️ **Informações de Referência**
- [Limitações e Honestidade Total](#-limitações-e-honestidade-total)
- [Licença e Responsabilidade](#-licença-e-responsabilidade)
- [Agradecimentos](#-agradecimentos)


---

## 🎯 **O Que Este Sistema Faz**

### 🔮 **Previsão Inteligente**
- Analisa **VALE3.SA**, **ITSA4.SA** e **TAEE11.SA** em tempo real
- Calcula a **probabilidade de alta** para o próximo dia útil
- Dá um veredito claro: **🟢 OPORTUNIDADE** ou **🟡 OBSERVAR**

### 🧠 **Explicação Transparente** 
- Mostra **exatamente** quais fatores influenciaram a decisão
- Gráficos SHAP que revelam o "raciocínio" da IA
- Sem caixas-pretas: você entende cada passo

### 📊 **Validação Rigorosa**
- **Score de Robustez** (0-9) baseado em performance histórica
- Testa o modelo em dados que ele nunca viu antes
- Mostra taxa de acerto dos últimos 30 dias

### 🩺 **Monitoramento de Saúde**
- Detecta se o mercado mudou (Data Drift)
- Compara condições atuais com dados de treinamento
- Avisa quando o modelo pode estar "desatualizado"

### 📈 **Análise Completa de Performance**
- Simula como seria operar seguindo os sinais
- Calcula Sharpe Ratio, Drawdown, Taxa de Acerto
- Compara performance com o IBOVESPA

---

## 🎭 **A História Por Trás do Projeto**

> *"E se você tivesse um assistente que analisasse milhares de dados do mercado em segundos, nunca se cansasse, e ainda explicasse cada decisão de forma cristalina?"*

Imagine que você está diante da tela, observando os gráficos de **VALE3**, **ITSA4** ou **TAEE11**. O mercado está volátil, as notícias se contradizem, e você precisa tomar uma decisão. **Comprar? Vender? Esperar?**

Foi exatamente essa angústia que deu origem a este projeto. Não queríamos criar mais um "robô trader" que promete lucros mágicos. Queríamos algo diferente: **um parceiro inteligente e honesto**.

### 🤔 **O Problema Real**

Todo dia, milhões de pessoas enfrentam o mesmo dilema:
- 📊 **Sobrecarga de informação**: Gráficos, indicadores, notícias... por onde começar?
- 🎲 **Decisões emocionais**: O medo e a ganância nublam o julgamento
- ⏰ **Falta de tempo**: Quem tem horas para analisar cada movimento?
- 🔍 **Falta de método**: Como separar o sinal do ruído no mercado?

### 💡 **Nossa Solução**

Criamos um sistema que combina:
- 🧠 **Inteligência Artificial** (LightGBM) para detectar padrões
- 🔬 **Metodologia científica** para validar cada previsão  
- 🎯 **Transparência total** - você vê exatamente como chegamos a cada conclusão
- 📱 **Interface simples** que qualquer pessoa pode usar

**Resultado?** Um assistente que analisa 18+ indicadores técnicos em segundos e te diz: *"Olha, baseado nos últimos 3 anos de dados, existe uma oportunidade interessante aqui. Deixe-me te mostrar o porquê..."*

---
## 🚀 **Instalação Rápida**

### 📋 **Pré-requisitos**
- Python 3.9 ou superior
- 5 minutos do seu tempo ⏰

### ⚡ **Instalação em 3 Passos**

```bash
# 1️⃣ Clone o projeto
git clone https://github.com/seu-usuario/analise-preditiva-ativos.git
cd analise-preditiva-ativos

# 2️⃣ Instale as dependências  
pip install -r app/requirements.txt

# 3️⃣ Treine os modelos (IMPORTANTE - faça isso primeiro!)
python app/train.py
```

### 🎮 **Como Usar**

```bash
# Inicie o dashboard
python app/run.py

# Abra seu navegador em: http://localhost:8501
```

**Pronto!** 🎉 Agora você tem seu assistente de análise funcionando!

---

## 🎛️ **Personalização e Configuração**

Tudo pode ser ajustado no arquivo `app/src/config/params.py`:

### 📊 **Ativos Analisados**
```python
TICKERS = ["ITSA4.SA", "VALE3.SA", "TAEE11.SA"]
# Adicione outros ativos da B3 aqui!
```

### ⏰ **Período de Análise**  
```python
PERIODO_DADOS = "3y"  # 3 anos de histórico
# Pode usar: "1y", "2y", "5y", "6mo", etc.
```

### 🎯 **Estratégia de Risco**
```python
ATR_FACTORS = {
    "VALE3.SA": (1.3, 0.8),  # (Take Profit, Stop Loss)
    "ITSA4.SA": (1.2, 0.8),
    # Ajuste conforme seu perfil de risco
}
```

### 🧠 **Inteligência do Modelo**
```python
N_FEATURES_A_SELECIONAR = 18  # Quantos indicadores usar
OPTUNA_N_TRIALS = 100        # Quantas tentativas de otimização
```

---

## 📊 **Entendendo os Resultados**

### 🎯 **Score de Robustez (0-9)**

Este é nosso "selo de qualidade". Calculamos baseado em 3 pilares:

**🏆 Risco-Retorno (Sharpe Ratio)**
- Sharpe > 1.0: +3 pontos (Excelente!)
- Sharpe > 0.3: +2 pontos (Bom)  
- Sharpe > -0.1: +1 ponto (Aceitável)

**🎯 Qualidade Preditiva (F1-Score)**
- F1 > 65%: +3 pontos (Ótima precisão)
- F1 > 55%: +2 pontos (Boa precisão)
- F1 > 50%: +1 ponto (Razoável)

**📈 Relevância Estatística (Número de Trades)**
- Média > 8 trades/fold: +3 pontos (Muito ativo)
- Média > 4 trades/fold: +2 pontos (Moderado)
- Média > 2.5 trades/fold: +1 ponto (Seletivo)

**🏅 Interpretação Final:**
- **7-9 pontos**: 🟢 Alta Confiança
- **4-6 pontos**: 🟡 Média Confiança  
- **0-3 pontos**: 🔴 Baixa Confiança

### 📈 **Métricas de Performance**

**Sharpe Ratio**: Retorno ajustado ao risco
- \> 1.0 = Excelente
- \> 0.5 = Bom
- \> 0 = Positivo

**Taxa de Acerto**: % de operações lucrativas
- \> 60% = Muito bom
- \> 50% = Bom
- < 50% = Precisa melhorar

**Max Drawdown**: Maior perda consecutiva
- < 10% = Excelente controle de risco
- < 20% = Aceitável
- \> 20% = Atenção necessária

---

## 🔍 **Casos de Uso Reais**

### 👨‍💼 **Para o Investidor Pessoa Física**
*"Tenho R$ 10.000 para investir. Vale a pena comprar VALE3 hoje?"*

**O sistema responde:**
- 🟢 **OPORTUNIDADE** (Probabilidade: 68%)
- **Explicação**: RSI em zona de sobrevenda + volume acima da média
- **Histórico**: Últimas 8 oportunidades similares, 6 foram lucrativas
- **Risco**: Score 6/9 - confiança média

### 📊 **Para o Trader Ativo**
*"Preciso de sinais confiáveis para day trade"*

**O sistema oferece:**
- Análise diária automática de 3 ativos
- Validação dos últimos 30 sinais
- Detecção de mudanças no padrão do mercado
- Explicação detalhada de cada decisão

### 🎓 **Para o Estudante/Pesquisador**
*"Quero entender como ML funciona em finanças"*

**O sistema ensina:**
- Metodologia completa de validação temporal
- Código aberto para estudar e modificar
- Explicações SHAP para entender as decisões
- Comparação de diferentes estratégias

---

## 🧬 **Como Funciona Por Dentro**

### 🎯 **1. A "Tripla Barreira" - Nossa Metodologia Secreta**

Imagine que você compra uma ação hoje. Definimos 3 cenários:
- 📈 **Barreira de Lucro**: Se subir X%, vendemos com lucro
- 📉 **Barreira de Perda**: Se cair Y%, vendemos no prejuízo  
- ⏰ **Barreira de Tempo**: Se nada acontecer em 5 dias, saímos

O **X** e **Y** não são fixos! Eles se adaptam à volatilidade de cada ação:
- VALE3 (mais volátil): barreiras mais largas
- ITSA4 (menos volátil): barreiras mais estreitas

### 🔬 **2. Validação "Walk-Forward" - O Teste da Vida Real**

Não testamos o modelo nos mesmos dados que ele aprendeu (isso seria "cola"!). 

Exemplo:
1. **2020-2021**: Modelo aprende
2. **2022**: Testamos (modelo nunca viu esses dados)
3. **2020-2022**: Modelo aprende novamente  
4. **2023**: Testamos novamente
5. E assim por diante...

É como se fosse um **simulador de tempo real** - o modelo só vê o futuro quando chega lá!

### 🧠 **3. 18 "Sensores" Analisando o Mercado**

O sistema monitora constantemente:

**📊 Momentum (Força do Movimento)**
- RSI: A ação está "cara" ou "barata"?
- Estocástico: Está em zona de sobrecompra/sobrevenda?
- Retornos: Como se comportou nos últimos dias?

**📈 Tendência (Direção Geral)**  
- Médias Móveis: Está acima ou abaixo da tendência?
- MACD: A força está aumentando ou diminuindo?

**🌊 Volatilidade (Nervosismo do Mercado)**
- Bandas de Bollinger: Está nos extremos?
- ATR: O mercado está calmo ou agitado?

**📦 Volume (Interesse dos Investidores)**
- Volume relativo: Há mais gente negociando que o normal?
- OBV: O dinheiro está entrando ou saindo?

**🏛️ Contexto (Como Está o Brasil)**
- Correlação com IBOVESPA: Está seguindo ou indo contra o mercado?
- IBOV vs Média: O país está otimista ou pessimista?

### 🎛️ **4. O "Cérebro" LightGBM**

Usamos um algoritmo chamado **LightGBM** - imagine um comitê de 1000 especialistas, cada um dando sua opinião:

- **Especialista 1**: "RSI baixo + volume alto = compra!"
- **Especialista 2**: "Mas a tendência está ruim..."  
- **Especialista 3**: "Porém o IBOV está forte..."
- **Decisão Final**: Média ponderada de todas as opiniões

### 🎯 **5. Calibração do "Gatilho"**

O modelo não diz apenas "sim" ou "não". Ele dá uma **probabilidade** (ex: 73.2%).

Mas quando consideramos uma "oportunidade"? 50%? 60%? 70%?

O sistema **testa automaticamente** diferentes valores e escolhe aquele que historicamente deu o melhor equilíbrio entre:
- **Precisão**: Quando diz "oportunidade", acerta?
- **Recall**: Consegue pegar a maioria das oportunidades reais?

---

## 🛠️ **Arquitetura Técnica**

### 📁 **Estrutura do Projeto**
```
app/
├── 🚀 run.py                    # Inicia a aplicação
├── 🎯 train.py                  # Treina os modelos
├── 📊 app.py                    # Dashboard principal
├── 📋 requirements.txt          # Dependências
└── src/                         # Código fonte
    ├── 🧠 models/              # Modelos de ML
    ├── 📈 data/                # Carregamento de dados
    ├── 🔍 backtesting/         # Análise de risco
    ├── 🎨 ui/                  # Interface do usuário
    ├── ⚙️ config/              # Configurações
    └── 📊 dados/               # Cache local (SQLite)
```

---
### 🔄 **Fluxo de Dados**

1. **📥 Coleta**: Yahoo Finance → Cache SQLite
2. **🔧 Processamento**: 18 indicadores técnicos
3. **🧠 Treinamento**: LightGBM + Optuna
4. **✅ Validação**: Walk-Forward + PurgedKFold
5. **💾 Persistência**: Modelos salvos (.joblib)
6. **📊 Interface**: Streamlit Dashboard
7. **🔄 Atualização**: Background thread (30min)

---

## 🧪 **Tecnologias Utilizadas**

Cada tecnologia foi cuidadosamente escolhida para resolver desafios específicos do projeto:

#### 🧠 **Machine Learning & IA**

**🧠 `LightGBM`** - *O Cérebro do Sistema*
- **Por quê?** Algoritmo de gradient boosting extremamente eficiente para dados tabulares
- **Vantagem**: Rápido, preciso e lida bem com features categóricas e numéricas
- **No projeto**: Modelo principal que processa os 18 indicadores técnicos

**⚡ `Optuna`** - *O Otimizador Inteligente*
- **Por quê?** Framework de otimização bayesiana para encontrar os melhores hiperparâmetros
- **Vantagem**: Muito mais eficiente que grid search, com early stopping automático
- **No projeto**: Maximiza o Sharpe Ratio durante o treinamento (100+ tentativas em minutos)

**🔍 `SHAP`** - *O Explicador Transparente*
- **Por quê?** Única biblioteca que explica decisões de ML de forma matematicamente rigorosa
- **Vantagem**: Mostra exatamente quanto cada feature contribuiu para a previsão
- **No projeto**: Gera os gráficos waterfall que revelam o "raciocínio" da IA

**🛠️ `Scikit-learn`** - *A Base Sólida*
- **Por quê?** Padrão da indústria para pré-processamento e métricas de ML
- **Vantagem**: APIs consistentes, bem documentadas e amplamente testadas
- **No projeto**: Scaling, seleção de features, validação cruzada e métricas

#### 📊 **Processamento de Dados Financeiros**

**📊 `Pandas`** - *O Manipulador de Séries Temporais*
- **Por quê?** Projetado especificamente para análise de dados financeiros
- **Vantagem**: Operações vetorizadas ultra-rápidas e índices temporais nativos
- **No projeto**: Processa milhares de candles OHLCV em milissegundos

**🔢 `NumPy`** - *O Motor de Cálculo*
- **Por quê?** Base matemática de todo o ecossistema científico Python
- **Vantagem**: Operações matriciais otimizadas em C, 100x mais rápido que Python puro
- **No projeto**: Cálculos de indicadores técnicos, métricas de risco e backtesting

**📈 `yFinance`** - *O Coletor de Dados*
- **Por quê?** API gratuita e confiável para dados do Yahoo Finance
- **Vantagem**: Dados históricos completos da B3 sem custos ou limitações
- **No projeto**: Download automático de OHLCV + dados do IBOVESPA

**💾 `SQLite`** - *O Cache Inteligente*
- **Por quê?** Banco embarcado, zero configuração, perfeito para cache local e principalmente por se tratar apenas de uma entrega de trabalho e não algo comercial 
- **Vantagem**: Evita downloads repetidos e permite operação offline de forma rápida
- **No projeto**: Armazena 3+ anos de dados históricos localmente

#### 🎨 **Interface & Experiência do Usuário**

**🎨 `Streamlit`** - *O Dashboard Moderno*
- **Por quê?** Transforma scripts Python em aplicações web profissionais
- **Vantagem**: Deploy instantâneo, componentes interativos nativos, cache automático
- **No projeto**: Interface completa com abas, métricas, gráficos e explicações

**📊 `Plotly`** - *A Visualização Interativa*
- **Por quê?** Gráficos financeiros profissionais com interatividade nativa
- **Vantagem**: Zoom, hover, legendas dinâmicas - experiência similar ao TradingView
- **No projeto**: Gráficos de preços, performance, equity curves e análises técnicas

**📈 `Matplotlib`** - *O Renderizador SHAP*
- **Por quê?** Integração nativa com SHAP para gráficos de explicabilidade
- **Vantagem**: Controle total sobre visualizações científicas e acadêmicas
- **No projeto**: Gráficos waterfall e matrizes de confusão

#### 🔧 **Infraestrutura & Performance**

**🔄 `Threading`** - *O Atualizador em Background*
- **Por quê?** Atualização de dados sem travar a interface do usuário
- **Vantagem**: Experiência fluida mesmo durante downloads de dados
- **No projeto**: Thread dedicada que atualiza dados a cada 30 minutos

**💾 `Joblib`** - *O Persistidor de Modelos*
- **Por quê?** Serialização otimizada para objetos científicos Python
- **Vantagem**: Carregamento ultra-rápido de modelos treinados (< 1 segundo)
- **No projeto**: Salva/carrega modelos LightGBM com todos os metadados

> 💡 **Filosofia de Escolha**: Priorizamos bibliotecas **maduras**, **bem documentadas** e **amplamente adotadas** pela comunidade científica, garantindo estabilidade e facilidade de manutenção.

---

## ⚠️ **Limitações e Honestidade Total**

### ❌ **O que este sistema NÃO é:**
- ❌ **Garantia de lucro**: Performance passada ≠ resultados futuros
- ❌ **Bola de cristal**: Não prevemos crashes ou eventos inesperados
- ❌ **Substituto para análise**: Sempre combine com fundamentalista
- ❌ **Conselho financeiro**: Somos uma ferramenta, não consultores

### ✅ **O que este sistema É:**
- ✅ **Ferramenta educacional**: Para aprender ML aplicado a finanças
- ✅ **Sistema de apoio**: Uma camada extra de informação
- ✅ **Plataforma de pesquisa**: Para testar estratégias
- ✅ **Código transparente**: Auditável e modificável

### 🛡️ **Uso Responsável**

**Antes de qualquer decisão:**
1. 📚 **Estude o ativo**: Fundamentos, setor, concorrência
2. 📊 **Analise o contexto**: Cenário macro, notícias, eventos
3. 💰 **Gerencie risco**: Nunca invista mais do que pode perder
4. 🎯 **Diversifique**: Não concentre em um único ativo
5. 👨‍💼 **Consulte profissionais**: CPA, CFP, analistas credenciados

---

## ⚖️ **Licença e Responsabilidade**

### 📄 **Licença MIT**
Este projeto é **open source** sob a licença MIT. Você pode:
- ✅ Usar comercialmente
- ✅ Modificar o código
- ✅ Distribuir
- ✅ Usar privadamente

**Apenas pedimos que:**
- 📝 Mantenha o aviso de copyright
- 📋 Inclua a licença MIT

### ⚠️ **Isenção de Responsabilidade**

```
🚨 IMPORTANTE: Este projeto é para fins EDUCACIONAIS e de PESQUISA.

❌ NÃO é consultoria financeira
❌ NÃO garante lucros
❌ NÃO substitui análise profissional

✅ Use por sua conta e risco
✅ Sempre consulte profissionais qualificados
✅ Nunca invista mais do que pode perder

Os desenvolvedores NÃO se responsabilizam por perdas financeiras.
```

---

## 🎉 **Agradecimentos**

### 🙏 **Inspirações e Referências**
- 📚 **"Advances in Financial Machine Learning"** - Marcos López de Prado
- 🏛️ **Quantitative Finance Community** - Pelas discussões valiosas
- 🐍 **Python Community** - Pelas ferramentas incríveis
- 📊 **Yahoo Finance** - Pelos dados gratuitos