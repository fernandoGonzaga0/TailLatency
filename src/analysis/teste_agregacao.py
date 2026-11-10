"""Verificação da agregação por cenário — Fase 6, item 1.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.teste_agregacao

Se aparecer "Agregação funcionando corretamente.", o IC e as tabelas estão
verificados. Usa execuções sintéticas em pasta temporária; roda em segundos.
"""

import math

import numpy as np

from src.analysis.agregacao import (
    COLUNAS_CENARIOS,
    COLUNAS_EXECUCOES,
    agregar_cenarios,
    conteudo_csv,
    gravar_csv,
    ic95_t,
    tabela_execucoes,
)
from src.analysis.carga import ler_indice
from src.analysis.metricas import METRICAS
from src.analysis.teste_carga import _gravar_bruto, _tabela_sintetica, _escrever_indice
import tempfile
from pathlib import Path

LIMIARES = {
    "regra": "x",
    "percentis": {"p90": 0.9, "p95": 0.95, "p99": 0.99},
    "limiares": [
        {"xi": 0.5, "limiar_p90_ms": 11.0, "limiar_p95_ms": 14.0, "limiar_p99_ms": 20.0},
    ],
}


def teste_ic95_valores_conhecidos():
    media, semi = ic95_t([1, 2, 3, 4, 5])
    # t(0,975; 4 gl) = 2,7764451; s = sqrt(2,5); semilargura = t * s / sqrt(5)
    esperado = 2.7764451051977987 * math.sqrt(2.5) / math.sqrt(5)
    assert media == 3.0
    assert abs(semi - esperado) < 1e-9
    assert abs(semi - 1.9632) < 1e-3
    media, semi = ic95_t([0.4, 0.4, 0.4])
    assert abs(media - 0.4) < 1e-12 and semi < 1e-12  # sem variação, IC de largura ~zero


def teste_ic95_exige_duas_replicacoes():
    try:
        ic95_t([1.0])
    except ValueError:
        return
    raise AssertionError("uma replicação só deveria falhar")


def teste_ic_cobre_a_media_verdadeira_na_frequencia_esperada():
    # replicações independentes N(10, 1): ~95% dos ICs com n = 5 devem cobrir 10
    rng = np.random.default_rng(11)
    cobre = 0
    n_ensaios = 4000
    for _ in range(n_ensaios):
        media, semi = ic95_t(rng.normal(10.0, 1.0, 5))
        cobre += abs(media - 10.0) <= semi
    frac = cobre / n_ensaios
    assert abs(frac - 0.95) < 0.015, frac  # com a normal (1,96) daria ~88%


def _execucoes_sinteticas():
    base = Path(tempfile.mkdtemp(prefix="teste_agregacao_"))
    raw = base / "raw"
    raw.mkdir()
    linhas = []
    for id_cenario, n_fan, c in (("N005_C04_xi0.50", 5, 4), ("N005_C08_xi0.50", 5, 8)):
        for rep in range(3):
            tabela = _tabela_sintetica(n=300, semente=100 * c + rep)
            if c == 8:  # contenção: acrescenta fila
                tabela["latencia_conjunto_ms"] += 5.0
                tabela["espera_media_ms"] = 1.0
            sha = _gravar_bruto(raw, id_cenario, rep, tabela)
            linhas.append({
                "id": id_cenario, "n_fan_out": n_fan, "concorrencia": c, "xi": 0.5,
                "replicacao": rep, "seed": rep, "n_requisicoes": 300, "sha256_bruto": sha,
                "violacao_p95": 0.0, "alteracoes_locais": False, "commit": "abc",
            })
    indice = base / "indice.csv"
    _escrever_indice(indice, linhas)
    return base, raw, ler_indice(indice)


def teste_tabela_por_execucao():
    base, raw, indice = _execucoes_sinteticas()
    execs = tabela_execucoes(indice, raw, LIMIARES)
    assert len(execs) == 6
    assert [(x["id"], x["replicacao"]) for x in execs] == sorted(
        (x["id"], x["replicacao"]) for x in execs)
    for x in execs:
        assert tuple(x) == COLUNAS_EXECUCOES
        assert x["violacao_p99"] <= x["violacao_p95"] <= x["violacao_p90"]
    # a replicação com fila viola mais que a sem fila (mesma semente-base não, mas +5 ms)
    sem = np.mean([x["violacao_p95"] for x in execs if x["concorrencia"] == 4])
    com = np.mean([x["violacao_p95"] for x in execs if x["concorrencia"] == 8])
    assert com > sem
    # espera média só no cenário com fila
    assert all(x["espera_media_chamada_ms"] == 0.0 for x in execs if x["concorrencia"] == 4)
    assert all(x["espera_media_chamada_ms"] == 1.0 for x in execs if x["concorrencia"] == 8)


def teste_agregar_cenarios():
    base, raw, indice = _execucoes_sinteticas()
    execs = tabela_execucoes(indice, raw, LIMIARES)
    cen = agregar_cenarios(execs, capacidade=4)
    assert len(cen) == 2
    assert [c["id"] for c in cen] == ["N005_C04_xi0.50", "N005_C08_xi0.50"]
    for c in cen:
        assert tuple(c) == COLUNAS_CENARIOS
        assert c["n_replicacoes"] == 3
    assert cen[0]["sem_contencao"] is True and cen[1]["sem_contencao"] is False
    valores = [x["p50_ms"] for x in execs if x["id"] == "N005_C04_xi0.50"]
    media, semi = ic95_t(valores)
    assert cen[0]["p50_ms_media"] == media and cen[0]["p50_ms_ic95"] == semi
    assert all(f"{m}_media" in cen[0] and f"{m}_ic95" in cen[0] for m in METRICAS)
    # capacidade maior muda a classificação
    assert agregar_cenarios(execs, capacidade=8)[1]["sem_contencao"] is True


def teste_agregar_recusa_parametros_diferentes_e_replicacao_unica():
    base, raw, indice = _execucoes_sinteticas()
    execs = tabela_execucoes(indice, raw, LIMIARES)
    mistura = [dict(x) for x in execs]
    mistura[1]["concorrencia"] = 99  # mesma id, parâmetro diferente
    try:
        agregar_cenarios(mistura, capacidade=4)
    except ValueError:
        pass
    else:
        raise AssertionError("parâmetros diferentes no mesmo cenário deveriam falhar")
    uma = [x for x in execs if x["replicacao"] == 0]
    try:
        agregar_cenarios(uma, capacidade=4)
    except ValueError:
        pass
    else:
        raise AssertionError("uma replicação por cenário deveria falhar")


def teste_csv_deterministico_e_relido():
    base, raw, indice = _execucoes_sinteticas()
    execs = tabela_execucoes(indice, raw, LIMIARES)
    cen = agregar_cenarios(execs, capacidade=4)
    a = gravar_csv(cen, COLUNAS_CENARIOS, base / "a.csv").read_bytes()
    b = gravar_csv(cen, COLUNAS_CENARIOS, base / "b.csv").read_bytes()
    assert a == b == conteudo_csv(cen, COLUNAS_CENARIOS)
    assert b"\r" not in a
    assert not list(base.glob("*.tmp"))  # gravação atômica não deixa temporário
    import csv, io
    linhas = list(csv.DictReader(io.StringIO(a.decode("utf-8"))))
    assert len(linhas) == 2
    assert float(linhas[0]["p99_ms_media"]) == cen[0]["p99_ms_media"]  # repr: releitura exata
    assert linhas[0]["sem_contencao"] == "true" and linhas[1]["sem_contencao"] == "false"


def main() -> None:
    testes = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"ok  {t.__name__}")
    print("Agregação funcionando corretamente.")


if __name__ == "__main__":
    main()
