"""Módulo de dados: organização e marcação do histórico de latência simulado, para uso direto pelas métricas de src/analysis/.

Estado atual (Fase 5 do checklist):

- `limiares.py`: metadados dos limiares de SLA (p90, p95 e p99 da chamada individual, em ms, por peso da cauda), gravados em `data/processed/limiares.json`.
- `marcacao.py`: lê o dado bruto de cada execução (sem limiar) e grava em `data/processed/` a mesma tabela com as colunas `viola_p90`, `viola_p95` e `viola_p99`, mais um registro por execução com os SHA-256 e a fração de violação.
- `teste_marcacao.py`: verificação (`python -m src.data.teste_marcacao`).

A marcação é tratada nas Fases 5 e 6: aqui ela gera o dado marcado e rastreável; na Fase 6 a análise aplica os limiares de novo sobre `latencia_conjunto_ms` (o bruto continua sem limiar) e confere que os dois caminhos concordam.
"""
