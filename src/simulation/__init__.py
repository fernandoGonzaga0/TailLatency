"""Módulo de simulação (SimPy) — Etapa 5 do TCC.

Objetivo
--------
Gerar, via simulação de eventos discretos, o histórico de latência de
chamadas a múltiplas APIs sob diferentes condições de fan-out e
concorrência. Essa saída alimenta diretamente a caracterização
estatística do fenômeno de amplificação de cauda (percentis p50/p95/p99,
fator de amplificação p99/p50, probabilidade de violação de SLA —
calculadas em src/analysis/).

Escopo
------
- As "APIs" são recursos/processos SimPy abstratos — não há chamadas de
  rede reais; a latência de cada chamada é sorteada estatisticamente
  (corpo lognormal + cauda GPD), não medida.
- NÃO inclui geração de features nem rotulagem para treino de modelo de
  machine learning. O componente preditivo (classificador de risco) foi
  removido do escopo do TCC em 22/09/2026 — ver a documentação do
  projeto para o histórico dessa decisão.

Entrada
-------
Arquivos de configuração de cenário em config/ (nº de APIs, grau de
fan-out, nível de concorrência, parâmetros da distribuição de latência
de cada API).

Saída
-----
Histórico bruto de latência em data/raw/: uma linha por chamada
(endpoint, horário simulado, carga do sistema no momento, latência
individual e latência do conjunto de chamadas concorrentes), consumido
depois por src/data/ (marcação por limiar de SLA/percentil) e por
src/analysis/ (cálculo das métricas).

Estado atual (22/09/2026)
--------------------------
Só a nota de escopo acima e um teste mínimo de ambiente
(sandbox_check.py) existem até agora. A modelagem dos recursos, a
geração de latência lognormal+GPD, os processos de fan-out e a
parametrização por cenário ainda serão implementados nos próximos
passos da Etapa 5.
"""
