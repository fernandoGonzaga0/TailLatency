# TCC — Caracterização do risco de cauda de latência em chamadas concorrentes a múltiplas APIs

Autor: Fernando Gonzaga

Simulação (SimPy) e caracterização estatística do risco de tail latency em chamadas concorrentes a múltiplas APIs, a partir do histórico de latência de cada endpoint.

> Nota (22/09/2026): a componente preditiva (classificador de risco) foi removida do escopo do projeto — ver a documentação do TCC para detalhes. O objetivo atual é caracterizar, via simulação, sob quais condições o risco de tail latency se intensifica.

## Objetivos e hipótese (07/10/2026)

**Objetivo geral:** caracterizar, por meio de simulação de eventos discretos, as condições em que chamadas concorrentes a múltiplas APIs sofrem amplificação de cauda de latência, analisando como o grau de fan-out, o nível de concorrência e o peso da cauda de latência das APIs afetam a probabilidade de a latência do conjunto de chamadas violar um limiar de referência, definido a partir da latência individual de cada chamada.

**Objetivos específicos:** (1) implementar, em Python com SimPy, um simulador de eventos discretos de APIs com latências lognormal com cauda GPD, fan-out e concorrência configuráveis; (2) gerar, de forma reprodutível, o histórico de latências sob uma grade de cenários (fan-out, concorrência e peso da cauda); (3) medir p50/p95/p99, o fator p99/p50 e a probabilidade de violação do limiar de referência (p95 da chamada individual, com p90 e p99 como sensibilidade); (4) validar o simulador contra a estatística de ordem em cenários sem contenção; (5) analisar o efeito do fan-out, da concorrência e do peso da cauda sobre o risco, verificando se as conclusões se mantêm entre os limiares e confrontando-as com a hipótese.

**Hipótese:** sem contenção, a probabilidade de a latência do conjunto violar o limiar de referência segue 1−q^N (q = 0,95 no p95) e cresce com o fan-out; com concorrência, fica acima desse valor e aumenta conforme o nível de concorrência.

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

## Decisões metodológicas (06/10/2026)

- **Limiar de violação de SLA:** p95 da latência de uma chamada individual, com referência fixa (uma API isolada, sem fan-out e sem contenção; não recalculado por cenário). O valor em ms é calculado no cenário base de cada combinação de parâmetros do gerador; com o corpo lognormal e o corte da spliced fixos, o p90 e o p95 não mudam com o peso da cauda e só o p99 varia.
- **Sensibilidade:** os limiares p90 e p99 também são avaliados, para verificar se as conclusões dependem da escolha do limiar.
- **O que é comparado:** a latência do conjunto (a mais lenta das N chamadas do fan-out) contra o limiar. Violação = conjunto acima do limiar.
- **APIs de um cenário:** todas compartilham os mesmos parâmetros de latência (lognormal + GPD). A heterogeneidade entre APIs fica como limitação e possível extensão.
- **Registro dos dados:** a simulação grava sempre as latências brutas (individual e do conjunto), sem limiar embutido. A marcação em `src/data/` cria uma coluna de violação por limiar (`viola_p90`, `viola_p95`, `viola_p99`), e os valores dos limiares em ms ficam em um arquivo de metadados.
- **Validação do módulo (cenário sem contenção):** a probabilidade de violação simulada deve bater com a teoria, 1−q^N com q = 0,90, 0,95 e 0,99, com o valor teórico dentro do intervalo de confiança de 95% da fração simulada. Se falhar, repetir com outra seed, sem ajustar a tolerância.
- **Robustez das conclusões:** o efeito deve ter a mesma direção, e os cenários devem ter a mesma ordem de risco, nos três limiares.
