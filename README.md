# 📈 Estratégia Quantitativa e Backtesting Causal (Walk-Forward) via SSA-SVR

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVR-orange.svg)](https://scikit-learn.org/)
[![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-green.svg)](LICENSE)

Pipeline quantitativo agnóstico e *end-to-end* para previsão da direção dos log-retornos diários no mercado acionário. A arquitetura combina **Análise de Espectro Singular (SSA)** para decomposição de ruído espectral com **Support Vector Regression (SVR)** sob um esquema causal de **Validação Walk-Forward**.

---

## 🔬 Design Técnico

* **Alvo em Log-Retornos:** Evita a previsão direta de preços absolutos (séries não-estacionárias), eliminando R² inflado e vieses de tendência.
* **Decomposição de Sinal Causal:** A decomposição matricial via SSA e o escalonamento (`StandardScaler`) são aplicados estritamente dentro das janelas históricas expandidas, eliminando qualquer risco de **Data Leakage**.
* **Validação Walk-Forward:** Re-treinamento diário com janela móvel de 500 dias, simulando fielmente as condições de execução do mercado em tempo real.
* **Agnóstico ao Ativo:** Arquitetura parametrizada para testar qualquer ativo de alta liquidez.

---

## 📊 Resultados Empíricos (Out-of-Sample: 299 Dias Úteis)

| Métrica | Estratégia do Modelo | Benchmark (Buy & Hold) | Detalhes / Significância |
| :--- | :---: | :---: | :--- |
| **Acurácia Direcional (DA)** | **57,19%** | - | 171 acertos em 299 operações |
| **Teste Binomial (p-valor)** | **0,0075** | - | **Estatisticamente Significativo (p < 0,01)** |
| **Retorno Acumulado** | **-0,91%** | -1,33% | Liquido de taxa de 0,05% por trade |
| **Índice de Sharpe Anualizado** | **0,07** | < 0 | Sharpe positivo em mercado de baixa |
| **MAPE do Preço Reconstruído** | **1,13%** | - | MAE: R$ 0,60 |

---

## 🖼️ Visualizações do Desempenho

### Reconstrução de Preço (A partir dos Log-Retornos Previstos)

### Curva de Capital Acumulado vs. Benchmark

---

## 🛠️ Arquitetura do Repositório

```text
├── data/              # Armazenamento de dados
├── docs/              # Gráficos em alta resolução do backtest
├── notebooks/         # Análises exploratórias (.ipynb)
├── src/               # Módulos do pipeline
│   ├── data.py        # Coleta de dados e geração de log-retornos
│   ├── metrics.py     # Métricas de regressão, direcionais e backtest
│   ├── model.py       # Loop Walk-Forward e treinamento do SVR
│   └── ssa.py         # Implementação da SSA via Matriz Hankel & SVD
├── main.py            # Script principal de execução
└── requirements.txt   # Dependências do projeto
```

---

## ⚡ Como Executar o Projeto

```bash
# 1. Clonar o repositório
git clone https://github.com/camargodiego/predicao-acoes-svr.git
cd predicao-acoes-svr

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Executar o pipeline completo
python main.py
```

---

## 📄 Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
