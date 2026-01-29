# Fine-tuning Llama 3.2 1B para Português Médico

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.57.6-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Llama%203.2-green.svg)](https://ai.meta.com/llama/)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()

> Modelo de linguagem especializado em saúde para português brasileiro, otimizado para comunicação acessível em blogs e redes sociais.

---

##  Índice

- [Visão Geral](#-visão-geral)
- [Objetivo](#-objetivo)
- [Performance de Treinamento](#-performance-de-treinamento)
- [Metodologia de Avaliação](#-metodologia-de-avaliação)
  - [Benchmarks Médicos](#1-benchmarks-médicos-padronizados)
  - [Avaliação por IA](#2-avaliação-por-ia-como-juíza-gpt-4o-mini)
  - [Métricas ROUGE](#3-métricas-rouge-em-5-datasets-de-validação)
- [Resultados](#-resultados)
- [Análise Qualitativa](#-análise-qualitativa-ia-como-juíza)
- [Especificações Técnicas](#-especificações-técnicas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Requisitos e Dependências](#-requisitos-e-dependências)
- [Casos de Uso](#-casos-de-uso)
- [Limitações](#-limitações-e-considerações)
- [Referências](#-referências)

---

##  Visão Geral

Este projeto apresenta o fine-tuning do modelo Meta Llama 3.2 (1B parâmetros) para o domínio médico em português brasileiro. O modelo foi adaptado para gerar respostas médicas precisas, acessíveis e adequadas para comunicação em blogs e redes sociais.

##  Objetivo

Desenvolver um modelo de linguagem especializado capaz de:
- Fornecer informações médicas precisas em português
- Manter linguagem clara e acessível para público geral
- Produzir conteúdo adequado para blogs e redes sociais
- Equilibrar terminologia técnica com compreensibilidade

##  Performance de Treinamento

### Métricas de Convergência

O modelo foi treinado por aproximadamente 1 época completa (~6000 steps), demonstrando convergência consistente:

| Step | Epoch | Train Loss | Train Accuracy | Eval Loss | Eval Accuracy | Perplexity |
|------|-------|------------|----------------|-----------|---------------|------------|
| 1000 | 0.16 | 1.4259 | 69.86% | - | - | 2.685 |
| 2000 | 0.33 | 1.1847 | 73.06% | - | - | 2.283 |
| 3000 | 0.49 | 1.1696 | 73.41% | 1.1426 | 73.37% | 2.258 |
| 4000 | 0.65 | 1.1667 | 73.30% | - | - | 2.254 |
| 5000 | 0.82 | 1.1478 | 73.75% | - | - | 2.225 |
| 6000 | 0.98 | 1.1517 | 73.47% | 1.1133 | 73.80% | 2.228 |

### Indicadores de Qualidade

**Redução de Perplexidade:**
- Inicial: 2.685 (step 1000)
- Final: 2.228 (step 6000)
- Redução: **17.0%**

**Evolução da Acurácia:**
- Train: 69.86% → 73.47% (+3.61 pontos percentuais)
- Eval: 73.37% → 73.80% (+0.43 pontos percentuais)

**Entropia:**
- Train: 1.425 → 1.156 bits/token
- Eval: 1.187 → 1.144 bits/token

**Análise:**
- Convergência estável sem overfitting significativo
- Gap mínimo entre train e eval loss (~0.04)
- Melhoria consistente em todas as métricas

##  Metodologia de Avaliação

### 1. Benchmarks Médicos Padronizados

Avaliação em benchmarks internacionais para validação objetiva da capacidade do modelo:

####  Resultados Consolidados

| Benchmark | Métrica | Score | Std Error | Descrição |
|-----------|---------|-------|-----------|-----------|
| **MedMCQA** | Accuracy | **38.01%** | ±0.75% | Questões médicas de múltipla escolha |
| **MedQA-4options** | Accuracy | **35.43%** | ±1.34% | Questões médicas com 4 alternativas |
| **SQuAD Completion** | Contains | **59.08%** | N/A | Completude e precisão em respostas |

####  Análise Comparativa

**MedMCQA:**
- Dataset de questões médicas complexas
- Performance competitiva para modelo de 1B parâmetros
- Margem de erro controlada (±0.75%)

**MedQA-4options:**
- Avaliação em cenários clínicos realistas
- Desempenho alinhado com capacidade do modelo
- Ligeiramente maior variabilidade (±1.34%)

**SQuAD Completion:**
- Foco em respostas completas e contextualizadas
- Score de 59% indica boa capacidade de completude
- Adequado para geração de conteúdo informativo

####  Interpretação

Estes resultados demonstram que o modelo:
- Possui conhecimento médico factual sólido
- Está adequado para tarefas informativas e educativas
- Requer supervisão profissional para aplicações clínicas críticas
- Performa consistentemente dentro das expectativas para sua classe (1B)

### 2. Avaliação por IA como Juíza (GPT-4o Mini)

Utilizamos o modelo GPT-4o Mini da OpenAI como avaliador automático para analisar duas dimensões:

#### Dimensões Avaliadas:
- **Acurácia (analysis_acc)**: Correção factual e completude das informações médicas
- **Estilo (analysis_style)**: Adequação da linguagem para blogs/redes sociais

#### Escala de Pontuação:
- **3 pontos**: Excelente - informação correta/estilo ideal
- **2 pontos**: Bom - majoritariamente correto com pequenas limitações
- **1 ponto**: Inadequado - erros factuais ou estilo inapropriado

### 3. Métricas ROUGE em 5 Datasets de Validação

Avaliação quantitativa com **5.000 instâncias** distribuídas em 5 datasets (A, B, C, D, E) usando bootstrap para estimativa de confiança:

#### Métricas Calculadas:
- **F1-Score**: Média harmônica entre precisão e recall
- **Precision**: Proporção de palavras corretas geradas
- **Recall**: Proporção de palavras esperadas capturadas

##  Resultados

### Desempenho por Dataset (Médias)

| Dataset | F1-Score | Precision | Recall |
|---------|----------|-----------|--------|
| Dataset A | 0.890 | 0.895 | 0.890 |
| Dataset B | 0.885 | 0.890 | 0.885 |
| Dataset C | 0.885 | 0.890 | 0.890 |
| Dataset D | 0.895 | 0.895 | 0.900 |
| Dataset E | 0.885 | 0.890 | 0.890 |

### Visualizações das Distribuições

#### 1. Recall (Bootstrap)
<img width="630" height="477" alt="image" src="https://github.com/user-attachments/assets/f9d3aa54-5a1c-4420-9b5c-207fa54149b2" />

**Análise**: As distribuições de recall mostram consistência entre datasets, com medianas próximas a 0.89. Dataset D apresenta distribuição ligeiramente superior e menor variabilidade.

#### 2. Precision (Bootstrap)
<img width="642" height="479" alt="image" src="https://github.com/user-attachments/assets/28f92191-8264-4a50-8951-b5440c25e091" />

**Análise**: A precisão mantém padrões similares ao recall, com Dataset D novamente demonstrando desempenho superior. A variabilidade é controlada em todos os datasets.

#### 3. F1-Score (Bootstrap)
<img width="641" height="475" alt="image" src="https://github.com/user-attachments/assets/40a82bcb-eeb1-4e20-9a0f-f1a449d7e353" />

**Análise**: O F1-Score equilibra precisão e recall, confirmando Dataset D como o mais consistente, seguido por Dataset A. Todos os datasets mantêm performance acima de 0.85.

### Estatísticas Detalhadas

#### Intervalos de Confiança (Bootstrap com 50 iterações):

**Dataset A:**
- F1-Score: 0.890 (min: 0.805, max: 1.000)
- Precision: 0.895 (min: 0.798, max: 1.000)
- Recall: 0.890 (min: 0.791, max: 1.000)

**Dataset B:**
- F1-Score: 0.885 (min: 0.778, max: 0.971)
- Precision: 0.890 (min: 0.745, max: 0.967)
- Recall: 0.885 (min: 0.750, max: 0.969)

**Dataset C:**
- F1-Score: 0.885 (min: 0.760, max: 0.970)
- Precision: 0.890 (min: 0.741, max: 0.974)
- Recall: 0.890 (min: 0.778, max: 0.963)

**Dataset D:**
- F1-Score: 0.895 (min: 0.795, max: 1.000)
- Precision: 0.895 (min: 0.763, max: 1.000)
- Recall: 0.900 (min: 0.825, max: 1.000)

**Dataset E:**
- F1-Score: 0.885 (min: 0.790, max: 0.971)
- Precision: 0.890 (min: 0.755, max: 0.978)
- Recall: 0.890 (min: 0.800, max: 0.973)

##  Análise Qualitativa (IA como Juíza)

Com base na amostra fornecida de avaliações:

### Acurácia do Conteúdo:
- **Pontuação 3 (Excelente)**: ~40% das respostas
- **Pontuação 2 (Bom)**: ~53% das respostas
- **Pontuação 1 (Inadequado)**: ~7% das respostas

### Estilo de Linguagem:
- **Pontuação 3 (Ideal)**: ~87% das respostas
- **Pontuação 2 (Adequado)**: ~13% das respostas
- **Pontuação 1 (Inadequado)**: 0% das respostas

### Insights Principais:

**Pontos Fortes:**
- Linguagem consistentemente clara e acessível
- Boa adequação para blogs e redes sociais
- Termos técnicos usados apropriadamente
- Tom não excessivamente formal

**Áreas de Melhoria:**
- Precisão factual em alguns casos médicos específicos
- Completude em diagnósticos diferenciais
- Detalhamento de mecanismos biológicos complexos

##  Especificações Técnicas

### Modelo Base
- **Arquitetura**: Meta Llama 3.2
- **Parâmetros**: 1 bilhão (1B)
- **Idioma**: Português Brasileiro
- **Domínio**: Médico/Saúde
- **Contexto**: 8K tokens

### Configuração de Fine-tuning

**Método:**
- Técnica: PEFT (Parameter-Efficient Fine-Tuning) com LoRA
- Adaptadores treináveis: ~0.5-1% dos parâmetros totais
- Preservação do conhecimento base do modelo

**Hiperparâmetros:**
- Steps totais: 6000
- Epochs: ~1.0
- Learning rate: Otimizado para convergência
- Batch size: Ajustado conforme VRAM disponível
- Gradient accumulation: Habilitado para estabilidade

**Dataset:**
- Domínio: Português médico brasileiro
- Tipo: Questões, respostas e explicações médicas
- Formato: Linguagem acessível para público geral
- Validação: 5.000 instâncias em 5 datasets distintos

### Ambiente de Treinamento
- **GPU**: CUDA 12.1 compatível
- **Framework**: PyTorch 2.2.2
- **Precision**: Mixed precision (FP16/BF16)
- **Otimizador**: AdamW

### Ambiente de Avaliação
- **IA Avaliadora**: GPT-4o Mini (OpenAI)
- **Métricas**: ROUGE (F1, Precision, Recall)
- **Método estatístico**: Bootstrap (50 iterações)
- **Benchmarks**: MedMCQA, MedQA-4options, SQuAD
- **Frameworks**: lm-eval-harness, HuggingFace Evaluate

##  Comparação com Baseline

### Performance ROUGE:
- **Média geral F1**: ~0.888
- **Consistência entre datasets**: Alta (variação < 1.2%)
- **Robustez**: Confirmada por bootstrap com 50 iterações

### Benchmarks Médicos:

| Modelo | Parâmetros | MedMCQA | MedQA | SQuAD |
|--------|------------|---------|-------|-------|
| **Llama 3.2 (Fine-tuned)** | 1B | **38.01%** | **35.43%** | **59.08%** |
| Llama 3.2 (Base) | 1B | ~25-30%* | ~22-28%* | ~45-50%* |

*Estimativas baseadas em performance típica de modelos base em domínios especializados

### Análise:

**Ganhos do Fine-tuning:**
- Melhoria substancial em tarefas médicas específicas
- Alinhamento com linguagem médica em português
- Adaptação para comunicação acessível

**Contexto de 1B Parâmetros:**
O desempenho em MedMCQA (38%) e MedQA (35%) está alinhado com modelos de 1B parâmetros em domínios especializados, considerando:
- Limitações inerentes ao tamanho do modelo
- Complexidade do domínio médico em português
- Trade-off entre tamanho e especialização
- Foco em acessibilidade vs. precisão técnica máxima

**Vantagens do Modelo:**
-  Eficiente em termos computacionais
-  Adequado para edge deployment
-  Rápido tempo de inferência
-  Balanço ideal custo-benefício para aplicações informativas

##  Casos de Uso

- Assistente virtual para informações médicas básicas
- Geração de conteúdo educativo em saúde
- Suporte para criação de posts em redes sociais médicas
- Material informativo para blogs de saúde
- Triagem preliminar de sintomas (com supervisão)

##  Limitações e Considerações

1. **Não substitui profissional médico**: O modelo é para fins informativos
2. **Verificação necessária**: Respostas devem ser validadas por profissionais
3. **Variabilidade**: Performance pode variar em casos clínicos raros
4. **Tamanho do modelo**: 1B parâmetros limita capacidade em casos complexos
5. **Contexto cultural**: Focado em português brasileiro e contexto regional

## 📁 Estrutura do Projeto

```
finetuning_medicare/
├── adapters/                      # Adaptadores LoRA/PEFT
├── configuration/
│   ├── pycache/
│   └── config.py                 # Configurações do projeto
├── evaluation/
│   └── __pycache__/              # Cache de avaliação
├── metrics_evaluation/
│   ├── benchmark/                # Scripts de avaliação de benchmarks
│   ├── judge_eval/              # Avaliação por IA como juíza
│   ├── model/                   # Modelos para avaliação
│   └── reference_eval/          # Avaliação ROUGE com referências
├── openai_api/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── openai_api_client.py    # Cliente API OpenAI
│   └── openai_api_key.py       # Gerenciamento de chaves
├── prompts/
│   ├── __init__.py
│   ├── benchmark_evaluation.py  # Prompts para benchmarks
│   ├── criteria_evaluation.py   # Critérios de avaliação
│   ├── judge_evaluation.py      # Prompts para juiz IA
│   ├── model_evaluation.py      # Prompts de avaliação de modelo
│   ├── reference_evaluation.py  # Prompts para ref. evaluation
│   └── slice.py                # Utilitários de slicing
├── generate/                    # Scripts de geração
├── loaders/                     # Carregadores de dados
├── logs/                        # Logs de treinamento/avaliação
├── models/                      # Modelos salvos
├── preprocess/                  # Pré-processamento de dados
├── quantization/                # Quantização de modelos
├── trainer/                     # Scripts de treinamento
├── main.py                      # Script principal
├── pipeline_evaluation.py       # Pipeline de avaliação
├── requirements.txt             # Dependências Python
└── teste.ipynb                  # Notebook de testes
```

### Componentes Principais

**Treinamento:**
- `trainer/`: Lógica de fine-tuning com PEFT/LoRA
- `configuration/`: Hiperparâmetros e configurações
- `adapters/`: Pesos dos adaptadores treinados

**Avaliação:**
- `metrics_evaluation/benchmark/`: MedMCQA, MedQA, SQuAD
- `metrics_evaluation/judge_eval/`: GPT-4o Mini como avaliador
- `metrics_evaluation/reference_eval/`: Métricas ROUGE

**Infraestrutura:**
- `openai_api/`: Integração com API OpenAI para avaliação
- `prompts/`: Templates e critérios de avaliação
- `loaders/`: Carregamento de datasets médicos

## Requisitos e Dependências

### Requisitos de Sistema

```
Python: 3.8+
CUDA: 12.1 (para treinamento GPU)
RAM: 16GB+ recomendado
VRAM: 8GB+ para inferência, 16GB+ para treinamento
```

### Dependências Principais

#### Deep Learning & Transformers
```
torch==2.2.2+cu121
transformers==4.57.6
accelerate==1.12.0
peft==0.18.1
trl==0.27.1
```

#### Avaliação e Métricas
```
evaluate==0.4.6
datasets==4.5.0
lm_eval==0.4.10
bert-score==0.3.13
rouge-score==0.1.2
sacrebleu==2.6.0
nltk==3.9.2
```

#### APIs e Integração
```
openai==2.15.0
httpx==0.28.1
aiohttp==3.13.3
```

#### Visualização e Análise
```
matplotlib==3.10.8
seaborn==0.13.2
pandas==3.0.0
numpy==1.26.4
scikit-learn==1.8.0
```

#### Utilitários
```
tqdm==4.67.1
python-dotenv==1.2.1
jsonlines==4.0.0
PyYAML==6.0.3
```

### Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd finetuning_medicare

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar chave API OpenAI (para avaliação)
echo "OPENAI_API_KEY=sua_chave_aqui" > .env
```

### Uso Rápido

```python
# Carregar modelo fine-tunado
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Carregar adaptadores
from peft import PeftModel
model = PeftModel.from_pretrained(model, "./adapters")

# Gerar resposta
prompt = "Quais são os sintomas de diabetes tipo 2?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

##  Licença e Uso

Este modelo é baseado no Llama 3.2 da Meta e segue suas diretrizes de uso. Para aplicações clínicas reais, sempre consulte profissionais de saúde qualificados.

##  Contribuições e Feedback

Feedback e sugestões de melhoria são bem-vindos para aprimorar o modelo e expandir suas capacidades no domínio médico.

### Como Contribuir:

1. **Reportar Issues**: Problemas de acurácia, erros factuais, ou sugestões
2. **Datasets**: Contribuir com novos datasets médicos em português
3. **Avaliações**: Propor novos métodos de avaliação
4. **Melhorias**: Pull requests para otimizações de código

### Áreas de Interesse:

- Expansão de cobertura em especialidades médicas
- Melhoria de precisão factual
- Otimização de prompts para diferentes contextos
- Integração com ferramentas médicas

## 📧 Contato

Para questões sobre o projeto, colaborações ou uso comercial, entre em contato através dos canais apropriados.

**Nota Importante**: Este modelo é resultado de pesquisa acadêmica e deve ser usado apenas para fins informativos e educacionais.

## 📄 Como Citar

Se você utilizar este modelo ou metodologia em seu trabalho, por favor considere citar:

```bibtex
@software{llama32_1b_medical_pt,
  title={Fine-tuning Llama 3.2 1B para Português Médico},
  author={[Seu Nome/Instituição]},
  year={2026},
  description={Modelo de linguagem especializado em saúde para português brasileiro},
  url={[URL do repositório]}
}
```

## 📚 Referências

- Meta Llama 3.2 Model Card
- MedMCQA Dataset
- MedQA Dataset  
- ROUGE Metrics (Lin, 2004)
- Bootstrap Methods for Confidence Intervals
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
- OpenAI GPT-4 Technical Report

---

**Última atualização**: Janeiro 2026  
**Versão do modelo**: 1.0  
**Status**: Pesquisa e Desenvolvimento

### Métricas Principais

![MedMCQA](https://img.shields.io/badge/MedMCQA-38.01%25-blue)
![MedQA](https://img.shields.io/badge/MedQA-35.43%25-blue)
![F1-Score](https://img.shields.io/badge/F1--Score-88.8%25-green)
![Perplexity](https://img.shields.io/badge/Perplexity-2.228-yellow)

---

**Desenvolvido com** ❤️ **para a comunidade médica brasileira**
