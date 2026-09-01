# Otimização de Rotas para Entrega de Medicamentos e Insumos Hospitalares

**Tech Challenge — Fase 2 | Pós Tech FIAP — Inteligência Artificial para Desenvolvimento de Tecnologia**

Sistema de otimização de rotas (VRP — Vehicle Routing Problem) para entrega de medicamentos e insumos hospitalares, usando Algoritmo Genético combinado com uma LLM (Google Gemini) para geração de instruções operacionais, relatórios de eficiência e respostas a perguntas em linguagem natural.

---

## Índice

- [Visão geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como rodar](#como-rodar)
- [Testes automatizados](#testes-automatizados)
- [Comparativo de desempenho](#comparativo-de-desempenho)
- [Proposta de arquitetura em nuvem (AWS)](#proposta-de-arquitetura-em-nuvem-aws)
- [Autor](#autor)

---

## Visão geral

Dado um conjunto de pontos de entrega, uma frota de veículos com capacidade e autonomia limitadas, e entregas com diferentes níveis de prioridade (medicamentos críticos vs. insumos regulares), o sistema encontra rotas que minimizam a distância total percorrida, respeitando três restrições obrigatórias:

- **Capacidade de carga** por veículo
- **Autonomia** (distância máxima) por veículo
- **Prioridade de entregas** (medicamentos críticos são sequenciados o quanto antes em cada rota)

O projeto evoluiu a partir de uma base de código de Algoritmo Genético para o TSP (Problema do Caixeiro Viajante), expandida progressivamente até o VRP completo com múltiplos veículos e as três restrições acima.

## Funcionalidades

- ✅ Algoritmo Genético com cromossomo `(permutação, splits)` — pontos de corte evoluíveis, permitindo rotas de tamanho variável entre veículos
- ✅ Três restrições obrigatórias implementadas via penalização no fitness (capacidade, autonomia, prioridade)
- ✅ Integração com LLM (Google Gemini) para:
  - Geração de instruções para motoristas
  - Geração de relatórios de eficiência da frota
  - Respostas a perguntas em linguagem natural sobre as rotas
- ✅ Visualização das rotas otimizadas em mapa (matplotlib)
- ✅ Pipeline único conectando GA → mapa → LLM, sem passos manuais
- ✅ 20 testes automatizados (pytest) cobrindo operadores genéticos e cálculo de fitness
- ✅ Comparativo de desempenho entre diferentes configurações do GA
- ✅ Diagramas de arquitetura (software e proposta de nuvem AWS)

## Arquitetura

Fluxo de dados do sistema: dados de entrada alimentam o Algoritmo Genético, que produz a melhor solução encontrada; essa solução alimenta, em paralelo, a visualização em mapa e a integração com LLM (que gera as três saídas: instruções, relatório e respostas a perguntas).

![Arquitetura do sistema](diagrama_arquitetura.png)

## Estrutura do repositório

```
.
├── genetic_algorithm.py               # operadores genéticos base (crossover, mutação)
├── tsp.py                             # TSP original, com o bug de crossover corrigido
├── vrp.py                             # extensão para múltiplos veículos
├── vrp_capacity_fixed_splits.py       # capacidade de carga — v1 (limitação conhecida, mantida como registro)
├── vrp_capacity.py                    # capacidade de carga — v2, splits evoluíveis (corrigida)
├── vrp_capacity_autonomy.py           # capacidade + autonomia
├── vrp_capacity_autonomy_priority.py  # as três restrições obrigatórias combinadas
├── vrp_llm.py                         # integração com a API do Gemini
├── vrp_visualization.py               # geração do mapa de rotas
├── vrp_pipeline.py                    # pipeline completo: GA + mapa + LLM
├── vrp_experiments.py                 # comparativo entre configurações do GA
├── vrp_architecture_diagram.py        # gera o diagrama de arquitetura de software
├── test_vrp.py                        # 20 testes automatizados (pytest)
├── requirements.txt
├── .python-version                    # 3.11.9 (via pyenv)
├── relatorio_tecnico_fase2.docx       # relatório técnico completo
├── roteiro_video.md                   # roteiro do vídeo de demonstração
└── documentacao_tech_challenge_2.md   # registro detalhado de decisões técnicas e execuções
```

## Como rodar

### Pré-requisitos

- Python **3.11+** (o pacote `google-genai` exige 3.9 ou superior — recomendado usar [pyenv](https://github.com/pyenv/pyenv) se seu sistema tiver uma versão mais antiga por padrão)
- Uma chave de API gratuita do Google Gemini, obtida em [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (não exige cartão de crédito)

### Instalação

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configurar a chave da API

```bash
export GEMINI_API_KEY="sua-chave-aqui"        # Mac/Linux
$env:GEMINI_API_KEY="sua-chave-aqui"          # Windows (PowerShell)
```

Sem a chave configurada, o sistema roda em **modo dry-run**: mostra os prompts que seriam enviados à API, sem chamá-la de verdade — útil para revisar a lógica sem custo.

### Rodar o pipeline completo

```bash
python vrp_pipeline.py
```

Isso roda o Algoritmo Genético, gera o mapa de rotas (`rotas_otimizadas.png`) e chama a LLM para instruções, relatório e uma pergunta de exemplo.

### Rodar componentes individualmente

```bash
python vrp_capacity_autonomy_priority.py   # só o GA, com as três restrições
python vrp_visualization.py                # exemplo isolado de visualização
python vrp_llm.py                          # exemplo isolado de integração com LLM
python vrp_experiments.py                  # comparativo de configurações do GA
```

## Testes automatizados

```bash
pytest test_vrp.py -v
```

20 testes cobrindo `order_crossover`, `mutate`, splits evoluíveis, cálculo de demanda/distância e as três penalidades de fitness (capacidade, autonomia, prioridade). Um dos testes documenta, como teste de regressão, o comportamento do bug histórico do crossover corrigido durante o desenvolvimento.

## Comparativo de desempenho

Cinco configurações do GA foram comparadas (população e taxa de mutação), medindo fitness final e velocidade de convergência:

![Comparativo de configurações do GA](comparativo_configuracoes_ga.png)

A configuração com população maior (250) obteve o melhor resultado tanto em fitness final quanto em velocidade de convergência. Detalhes completos no [relatório técnico](relatorio_tecnico_fase2.docx).

## Proposta de arquitetura em nuvem (AWS)

Item opcional do enunciado: uma proposta de arquitetura serverless para rodar o pipeline na AWS (não implementada, apenas documentada):

![Proposta de arquitetura AWS](diagrama_arquitetura_aws_v2.jpg)

EventBridge dispara o pipeline, Step Functions orquestra as etapas, Lambda executa o GA e chama a API do Gemini, S3 armazena os dados em cada etapa, Secrets Manager protege a chave da API, e CloudWatch monitora a execução.

## Autor

**Leticia** — RM: [completar]
Pós Tech FIAP — Inteligência Artificial para Desenvolvimento de Tecnologia
