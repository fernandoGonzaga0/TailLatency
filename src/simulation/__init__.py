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

Estado atual (22/10/2026)
--------------------------
Passo 1 (nota de escopo e verificação do ambiente, sandbox_check.py), passo 2
(modelagem dos recursos, recursos.py, verificado por teste_recursos.py), passo 3
(gerador de latência lognormal + GPD, latencia.py, verificado por
teste_latencia.py e validar_latencia.py) e passo 4 (processos de requisição e
fan-out, requisicoes.py, verificado por teste_requisicoes.py) concluídos.
Decisões do passo 2: número de APIs por cenário = grau de fan-out N; capacidade
por API configurável, padrão 1; fila FIFO ilimitada. Decisões do passo 3: corte
da spliced em p_corte = 0,95 e scale da GPD por continuidade da densidade
(escolha do TCC, não do Kyung); peso da cauda controlado só pelo xi. Decisões do
passo 4: concorrência em sistema fechado, com C requisições em andamento e think
time zero (precedente: Tennage et al., 2019; Harchol-Balter, 2013, seção 2.6);
a latência do conjunto é o máximo das N latências individuais.
Passo 5 (parametrização por cenário, cenarios.py e config/grade.toml, verificado
por teste_cenarios.py) concluído em 22/10/2026: capacity = 4 fixa; grade de 5
fan-outs x 6 níveis de C x 4 pesos de cauda = 120 cenários; 5 replicações de
10.000 requisições com 500 de aquecimento; seeds derivadas de [seed_base,
cenário, replicação]; configuração em TOML (tomllib, sem dependência nova).
Passo 6 (reprodutibilidade, manifesto.py, verificado por teste_manifesto.py)
concluído em 22/10/2026: cada execução (cenário x replicação) grava um manifesto
JSON com cenário, seed, limiares em ms, hash da grade, versões, commit do git e
resumo do resultado; verificar_manifesto refaz a execução e compara com
tolerância.
Passo 7 (formato de saída, saida.py, verificado por teste_saida.py) concluído em
22/10/2026: uma linha por requisição (não por chamada), em CSV comprimido
(.csv.gz), um arquivo por execução ao lado do manifesto, com floats em precisão
completa e gravação determinística; o manifesto (versão 2) registra arquivo,
número de linhas e SHA-256 do conteúdo, e a verificação confere o arquivo. Cerca
de 0,4 MB por execução (cerca de 240 MB nas 600). A execução da grade (Fase 5)
ainda será implementada.
Passo 8 (validação do módulo, validacao.py, ferramentas verificadas por
teste_validacao.py) concluído em 22/10/2026: sem contenção (C <= capacidade) a
simulação é comparada com a teoria 1 - q^N (q = 0,90, 0,95 e 0,99), com os
percentis da chamada e do conjunto (quantil p^(1/N)), com IC de Wilson de 95%;
teste que falha é refeito uma vez com outra seed_base, sem mudar a tolerância.
"""
