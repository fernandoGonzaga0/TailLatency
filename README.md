# TCC — Previsão de risco de cauda de latência em chamadas concorrentes a múltiplas APIs

Autor: Fernando Gonzaga

Simulação (SimPy) e caracterização estatística do risco de tail latency em chamadas concorrentes a múltiplas APIs, a partir do histórico de latência de cada endpoint.

> Nota (22/09/2026): a componente preditiva (classificador de risco) foi removida do escopo do projeto — ver a documentação do TCC para detalhes. O objetivo atual é caracterizar, via simulação, sob quais condições o risco de tail latency se intensifica.

## Estrutura do projeto

- `src/simulation/` — simulação de eventos discretos (SimPy): recursos/processos representando as APIs, geração de latência (corpo lognormal + cauda GPD), parametrização de fan-out e concorrência.
- `src/data/` — organização e marcação do histórico de latência simulado (violação de limiar, quando útil para as métricas).
- `src/analysis/` — cálculo das métricas de avaliação (percentis, fator de amplificação de cauda, probabilidade de violação de SLA).
- `config/` — arquivos de configuração por cenário (nº de APIs, grau de fan-out, nível de concorrência).
- `notebooks/` — explorações rápidas e descartáveis em Jupyter, sem misturar com o código do pipeline principal.
- `data/raw/` — saída bruta das simulações.
- `data/processed/` — histórico de latência processado, pronto para análise.
- `results/` — métricas, gráficos e saídas da análise.

## Ambiente

Ambiente virtual local (`.venv`) com dependências fixadas em `requirements.txt`.
