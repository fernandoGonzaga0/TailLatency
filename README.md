# TCC — Caracterização do risco de cauda de latência em chamadas concorrentes a múltiplas APIs

Autor: Fernando Gonzaga

Simulação (SimPy) e caracterização estatística do risco de tail latency em chamadas concorrentes a múltiplas APIs, a partir do histórico de latência de cada endpoint.

> Nota (22/09/2026): a componente preditiva (classificador de risco) foi removida do escopo do projeto — ver a documentação do TCC para detalhes. O objetivo atual é caracterizar, via simulação, sob quais condições o risco de tail latency se intensifica.

## Objetivos e hipótese (07/10/2026)

**Objetivo geral:** caracterizar, por meio de simulação de eventos discretos, as condições em que chamadas concorrentes a múltiplas APIs sofrem amplificação de cauda de latência, analisando como o grau de fan-out, o nível de concorrência e o peso da cauda de latência das APIs afetam a probabilidade de a latência do conjunto de chamadas violar um limiar de referência, definido a partir da latência individual de cada chamada.

**Objetivos específicos:** (1) implementar, em Python com SimPy, um simulador de eventos discretos de APIs com latências lognormal com cauda GPD, fan-out e concorrência configuráveis; (2) gerar, de forma reprodutível, o histórico de latências sob uma grade de cenários (fan-out, concorrência e peso da cauda); (3) medir p50/p95/p99, o fator p99/p50 e a probabilidade de violação do limiar de referência (p95 da chamada individual, com p90 e p99 como sensibilidade); (4) validar o simulador contra a estatística de ordem em cenários sem contenção; (5) analisar o efeito do fan-out, da concorrência e do peso da cauda sobre o risco, verificando se as conclusões se mantêm entre os limiares e confrontando-as com a hipótese.

**Hipótese:** sem contenção, a probabilidade de a latência do conjunto violar o limiar de referência segue 1−q^N (q = 0,95 no p95) e cresce com o fan-out; com concorrência, fica acima desse valor e aumenta conforme o nível de concorrência.

## Estrutura do projeto

- `src/simulation/` — simulação de eventos discretos (SimPy): recursos com fila representando as APIs (`recursos.py`), geração de latência lognormal + GPD (`latencia.py`), requisições com fan-out em sistema fechado (`requisicoes.py`), grade de cenários (`cenarios.py`), manifesto de reprodutibilidade (`manifesto.py`), formato de saída (`saida.py`) e validação contra a teoria (`validacao.py`). Cada módulo tem um `teste_*.py` correspondente.
- `src/data/` — organização e marcação do histórico de latência simulado (violação de limiar, quando útil para as métricas).
- `src/analysis/` — cálculo das métricas de avaliação (percentis, fator de amplificação de cauda, probabilidade de violação de SLA).
- `config/` — `grade.toml`, a grade de cenários (fan-out, concorrência, peso da cauda, capacidade, replicações e seed).
- `notebooks/` — explorações rápidas e descartáveis em Jupyter, sem misturar com o código do pipeline principal.
- `data/raw/` — saída bruta das simulações: um `requisicoes_<id>_rep<k>.csv.gz` e um `manifesto_<id>_rep<k>.json` por execução (ignorada pelo git).
- `data/processed/` — histórico de latência processado, pronto para análise.
- `results/` — métricas, gráficos e saídas da análise, incluindo `validacao_modulo.json` (ignorada pelo git).

## Ambiente

Ambiente virtual local (`.venv`) com dependências fixadas em `requirements.txt` (Python 3.14 nos dois computadores do autor; a leitura do TOML usa o `tomllib` da biblioteca padrão).

## Estado do projeto (22/10/2026)

- **Concluídas:** definição do tema e dos objetivos, revisão bibliográfica (6 materiais), ambiente e **módulo de simulação (Fase 4, passos 1 a 8)**, publicado no commit `876e2bb`.
- **Próxima:** Fase 5 (script de execução da grade, marcação por limiar em `src/data/` e as 600 execuções definitivas), depois a análise (Fase 6) e a redação (Fase 7).
- **Conclusões:** a única já estabelecida é a validação do simulador, a primeira parte da hipótese. Sem contenção, a violação simulada bate com 1−q^N (q = 0,90, 0,95 e 0,99): nos 40 cenários sem contenção foram 360 testes, 14 fora do IC de 95% na primeira execução (3,9%, perto dos 5% esperados por acaso) e 0 após a repetição com outra seed. A segunda parte (com concorrência) e a análise do efeito do fan-out, da concorrência e do peso da cauda dependem dos dados da Fase 5 e da análise da Fase 6; o piloto aponta na direção da hipótese, mas não é resultado. Dois achados do piloto, a confirmar na Fase 6: o p99/p50 do conjunto cai com a concorrência, e há efeito de teto da violação com N grande.

## Como usar o módulo de simulação

Com o `.venv` ativado, a partir da raiz do projeto:

```
python -m src.simulation.cenarios --listar
python -m src.simulation.cenarios --rodar N010_C06_xi0.50 --replicacao 0
python -m src.simulation.manifesto --gerar N010_C06_xi0.50 --replicacao 0
python -m src.simulation.manifesto --verificar data/raw/manifesto_N010_C06_xi0.50_rep0.json
python -m src.simulation.validacao --rodar --processos 4
```

Testes (cada um termina com uma mensagem de sucesso): `python -m src.simulation.teste_recursos`, `teste_latencia`, `teste_requisicoes`, `teste_cenarios`, `teste_manifesto`, `teste_saida` e `teste_validacao`.

## Decisões metodológicas (06/10/2026)

- **Limiar de violação de SLA:** p95 da latência de uma chamada individual, com referência fixa (uma API isolada, sem fan-out e sem contenção; não recalculado por cenário). O valor em ms é calculado no cenário base de cada combinação de parâmetros do gerador; com o corpo lognormal e o corte da spliced fixos, o p90 e o p95 não mudam com o peso da cauda e só o p99 varia.
- **Sensibilidade:** os limiares p90 e p99 também são avaliados, para verificar se as conclusões dependem da escolha do limiar.
- **O que é comparado:** a latência do conjunto (a mais lenta das N chamadas do fan-out) contra o limiar. Violação = conjunto acima do limiar.
- **APIs de um cenário:** todas compartilham os mesmos parâmetros de latência (lognormal + GPD). A heterogeneidade entre APIs fica como limitação e possível extensão.
- **Registro dos dados:** a simulação grava sempre as latências brutas, sem limiar embutido, em uma linha por requisição (latência do conjunto, serviço do conjunto e maior e média das esperas das N chamadas), em `.csv.gz` por execução. A marcação em `src/data/` cria uma coluna de violação por limiar (`viola_p90`, `viola_p95`, `viola_p99`), e os valores dos limiares em ms ficam em um arquivo de metadados.
- **Validação do módulo (cenário sem contenção):** a probabilidade de violação simulada deve bater com a teoria, 1−q^N com q = 0,90, 0,95 e 0,99, com o valor teórico dentro do intervalo de confiança de 95% da fração simulada (IC de Wilson). Se falhar, repetir com outra seed, sem ajustar a tolerância. **Resultado (22/10/2026): aprovada**, ver o estado do projeto.
- **Robustez das conclusões:** o efeito deve ter a mesma direção, e os cenários devem ter a mesma ordem de risco, nos três limiares.

## Decisões da simulação (21 e 22/10/2026)

- **Latência de uma chamada:** espera na fila mais tempo de serviço. O serviço vem de uma distribuição em duas partes: lognormal (mediana de 10 ms, sigma de 0,5) até o percentil 95 (22,76 ms) e GPD acima, com peso da cauda ξ em {0,2; 0,5; 0,8; 1,1}. A escala da GPD (5,5170 ms) é fixada por continuidade da densidade no ponto de corte, escolha do TCC e não do artigo de referência.
- **Concorrência:** sistema fechado com C requisições em andamento e tempo de reflexão zero. Cada API tem 4 atendentes e fila FIFO ilimitada, então C até 4 não tem contenção.
- **Grade:** N em {1, 5, 10, 20, 50}, C em {1, 4, 6, 8, 10, 12} e ξ em {0,2; 0,5; 0,8; 1,1}, 120 cenários (40 sem contenção e 80 com), cada um com 5 replicações de 10.000 requisições medidas e 500 de aquecimento descartadas: 600 execuções. O IC de 95% é calculado entre replicações, porque com contenção as requisições de um laço são correlacionadas.
- **Reprodutibilidade:** seed de cada replicação derivada do cenário e da replicação (`SeedSequence`), manifesto JSON por execução (cenário, seed, versões, commit do git, hash da grade e resumo) e verificação por reexecução com tolerância. Para gerar os dados definitivos, a árvore do git deve estar limpa.
