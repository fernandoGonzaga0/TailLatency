"""Métricas por execução — Etapa 8 (Fase 6 do checklist, item 1).

Funções puras (sem leitura de arquivo) que calculam, a partir da tabela por
requisição de uma execução (cenário x replicação):

- percentis p50, p95 e p99 da latência do conjunto de chamadas;
- fator de amplificação de cauda p99/p50 (do conjunto);
- probabilidade de violação em cada limiar (p90, p95 e p99 da chamada
  individual isolada, valores em ms vindos de `limiares.json`);
- as mesmas violações calculadas sobre `servico_conjunto_ms` (máximo dos tempos
  de serviço, sem fila), que separam a violação causada pela cauda do serviço da
  causada pela fila (decisão de 10/11/2026);
- latência média e espera média por chamada.

Convenções (iguais às da Fase 4 e da marcação da Fase 5)
--------------------------------------------------------
- Percentil: `numpy.percentile` com interpolação linear (o mesmo de
  `manifesto.resumir_arrays`), para os valores recalculados baterem exatamente
  com o resumo gravado no manifesto e no índice.
- Violação: `valor > limiar`, estritamente acima; igual ao limiar não é violação.
- Só a probabilidade de violação depende do limiar; percentis e p99/p50 não.

Os nomes das métricas repetem os do índice (`p50_ms`, `p99_ms`,
`amplificacao_p99_p50`, `latencia_media_ms`, `espera_media_chamada_ms`,
`violacao_p90`, ...), para a conferência cruzada (item 2) comparar por nome.

Como rodar os testes (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.teste_metricas
"""

import numpy as np

NOMES_LIMIAR = ("p90", "p95", "p99")
COLUNAS_NECESSARIAS = (
    "latencia_conjunto_ms",
    "servico_conjunto_ms",
    "espera_media_ms",
)

METRICAS = (
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "amplificacao_p99_p50",
    "latencia_media_ms",
    "espera_media_chamada_ms",
    *(f"violacao_{n}" for n in NOMES_LIMIAR),
    *(f"violacao_servico_{n}" for n in NOMES_LIMIAR),
)


def percentil(valores, p: float) -> float:
    """Percentil p (de 0 a 100), interpolação linear (padrão do NumPy)."""
    x = np.asarray(valores, dtype=float)
    if x.size == 0:
        raise ValueError("não há valores para calcular o percentil")
    if not 0 <= p <= 100:
        raise ValueError(f"p deve estar entre 0 e 100, recebido {p!r}")
    return float(np.percentile(x, p))


def amplificacao(valores) -> float:
    """Fator de amplificação de cauda: p99 / p50 dos valores."""
    p50 = percentil(valores, 50)
    if p50 <= 0:
        raise ValueError("o p50 deve ser positivo para calcular p99/p50")
    return percentil(valores, 99) / p50


def contagem_acima(valores, limiar: float) -> int:
    """Quantos valores estão estritamente acima do limiar."""
    return int(np.count_nonzero(np.asarray(valores, dtype=float) > limiar))


def fracao_acima(valores, limiar: float) -> float:
    """Fração dos valores estritamente acima do limiar (probabilidade de violação)."""
    x = np.asarray(valores, dtype=float)
    if x.size == 0:
        raise ValueError("não há valores para calcular a fração")
    return float(np.mean(x > limiar))


def _validar_limiares(limiares_ms: dict) -> None:
    faltam = [n for n in NOMES_LIMIAR if n not in limiares_ms]
    if faltam:
        raise ValueError(f"faltam limiares: {faltam}")
    if not limiares_ms["p90"] <= limiares_ms["p95"] <= limiares_ms["p99"]:
        raise ValueError(f"limiares fora de ordem: {limiares_ms!r}")


def contagens_violacao(valores, limiares_ms: dict) -> dict:
    """{'p90': n, 'p95': n, 'p99': n}: contagem de valores acima de cada limiar."""
    _validar_limiares(limiares_ms)
    return {n: contagem_acima(valores, limiares_ms[n]) for n in NOMES_LIMIAR}


def metricas_execucao(tabela, limiares_ms: dict) -> dict:
    """Métricas de uma execução, a partir da tabela por requisição.

    `tabela` é um DataFrame (ou qualquer mapeamento de coluna para vetor) com as
    colunas de `COLUNAS_NECESSARIAS`; `limiares_ms` é {'p90','p95','p99'} em ms.
    Devolve um dicionário com `n_requisicoes` e as chaves de `METRICAS`.
    """
    faltam = [c for c in COLUNAS_NECESSARIAS if c not in tabela]
    if faltam:
        raise ValueError(f"colunas ausentes na tabela: {faltam}")
    _validar_limiares(limiares_ms)

    lat = np.asarray(tabela["latencia_conjunto_ms"], dtype=float)
    serv = np.asarray(tabela["servico_conjunto_ms"], dtype=float)
    espera = np.asarray(tabela["espera_media_ms"], dtype=float)
    if lat.size == 0:
        raise ValueError("a tabela não tem requisições")
    if not (lat.size == serv.size == espera.size):
        raise ValueError("as colunas têm tamanhos diferentes")

    p50 = percentil(lat, 50)
    p99 = percentil(lat, 99)
    if p50 <= 0:
        raise ValueError("o p50 deve ser positivo para calcular p99/p50")

    saida = {
        "n_requisicoes": int(lat.size),
        "p50_ms": p50,
        "p95_ms": percentil(lat, 95),
        "p99_ms": p99,
        "amplificacao_p99_p50": p99 / p50,
        "latencia_media_ms": float(np.mean(lat)),
        "espera_media_chamada_ms": float(np.mean(espera)),
    }
    for n in NOMES_LIMIAR:
        saida[f"violacao_{n}"] = fracao_acima(lat, limiares_ms[n])
    for n in NOMES_LIMIAR:
        saida[f"violacao_servico_{n}"] = fracao_acima(serv, limiares_ms[n])
    return saida
