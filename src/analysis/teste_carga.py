"""Verificação da carga rastreável — Fase 6, item 1.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.teste_carga

Se aparecer "Carga funcionando corretamente.", a leitura está verificada.
Cria execuções sintéticas em uma pasta temporária; roda em poucos segundos.
"""

import csv
import gzip
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.carga import (
    carregar_execucao,
    carregar_limiares,
    ler_indice,
    limiares_do_xi,
)
from src.simulation.saida import COLUNAS, nome_dados, sha256_conteudo


def _tabela_sintetica(n=200, semente=3):
    rng = np.random.default_rng(semente)
    lat = rng.lognormal(2.3, 0.5, n)
    return pd.DataFrame({
        "id_requisicao": np.arange(n),
        "laco": np.zeros(n, dtype=int),
        "inicio_ms": np.arange(n, dtype=float),
        "fim_ms": np.arange(n, dtype=float) + lat,
        "latencia_conjunto_ms": lat,
        "servico_conjunto_ms": lat,
        "espera_max_ms": np.zeros(n),
        "espera_media_ms": np.zeros(n),
    })


def _gravar_bruto(pasta: Path, id_cenario: str, replicacao: int, tabela) -> str:
    assert tuple(tabela.columns) == tuple(COLUNAS)
    texto = tabela.to_csv(index=False, lineterminator="\n", float_format=None)
    caminho = pasta / nome_dados(id_cenario, replicacao)
    with gzip.open(caminho, "wb") as f:
        f.write(texto.encode("utf-8"))
    return sha256_conteudo(caminho)


def _escrever_indice(caminho: Path, linhas: list[dict]) -> None:
    colunas = [
        "id", "n_fan_out", "concorrencia", "xi", "replicacao", "seed",
        "n_requisicoes", "sha256_bruto", "violacao_p95", "alteracoes_locais", "commit",
    ]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(colunas)
        for x in linhas:
            w.writerow([str(x[c]).lower() if isinstance(x[c], bool) else x[c] for c in colunas])


def _preparar():
    base = Path(tempfile.mkdtemp(prefix="teste_carga_"))
    raw = base / "raw"
    raw.mkdir()
    linhas = []
    for rep in (0, 1):
        tabela = _tabela_sintetica(semente=rep)
        sha = _gravar_bruto(raw, "N005_C04_xi0.50", rep, tabela)
        linhas.append({
            "id": "N005_C04_xi0.50", "n_fan_out": 5, "concorrencia": 4, "xi": 0.5,
            "replicacao": rep, "seed": 1000 + rep, "n_requisicoes": len(tabela),
            "sha256_bruto": sha, "violacao_p95": 0.05, "alteracoes_locais": False,
            "commit": "abc123",
        })
    indice = base / "indice.csv"
    _escrever_indice(indice, linhas)
    return base, raw, indice, linhas


def _limiares_json(base: Path) -> Path:
    dados = {
        "versao": 1,
        "regra": "viola_pXX = latencia_conjunto_ms > limiar_pXX_ms",
        "percentis": {"p90": 0.9, "p95": 0.95, "p99": 0.99},
        "limiares": [
            {"xi": 0.2, "limiar_p90_ms": 18.0, "limiar_p95_ms": 22.0, "limiar_p99_ms": 33.0},
            {"xi": 0.5, "limiar_p90_ms": 18.0, "limiar_p95_ms": 22.0, "limiar_p99_ms": 36.0},
        ],
    }
    caminho = base / "limiares.json"
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    return caminho


def teste_ler_indice_tipos():
    base, raw, indice, linhas = _preparar()
    lidas = ler_indice(indice)
    assert len(lidas) == 2
    x = lidas[0]
    assert isinstance(x["n_fan_out"], int) and isinstance(x["concorrencia"], int)
    assert isinstance(x["xi"], float) and x["xi"] == 0.5
    assert isinstance(x["seed"], int) and isinstance(x["n_requisicoes"], int)
    assert x["alteracoes_locais"] is False
    assert x["sha256_bruto"] == linhas[0]["sha256_bruto"]
    assert isinstance(x["violacao_p95"], float)


def teste_indice_invalido():
    base, raw, indice, linhas = _preparar()
    # coluna obrigatória faltando
    ruim = base / "ruim.csv"
    ruim.write_text("id,xi\nA,0.5\n", encoding="utf-8")
    try:
        ler_indice(ruim)
    except ValueError:
        pass
    else:
        raise AssertionError("índice sem colunas obrigatórias deveria falhar")
    # execução repetida
    _escrever_indice(base / "dup.csv", [linhas[0], linhas[0]])
    try:
        ler_indice(base / "dup.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("execução repetida deveria falhar")


def teste_carregar_execucao_confere_hash_e_linhas():
    base, raw, indice, linhas = _preparar()
    lida = ler_indice(indice)[0]
    tabela = carregar_execucao(lida, raw)
    assert len(tabela) == 200
    assert tabela["latencia_conjunto_ms"].dtype == np.float64
    original = _tabela_sintetica(semente=0)
    assert np.array_equal(tabela["latencia_conjunto_ms"].to_numpy(),
                          original["latencia_conjunto_ms"].to_numpy())  # releitura exata


def teste_detecta_arquivo_adulterado_ausente_e_linhas():
    base, raw, indice, linhas = _preparar()
    lida = ler_indice(indice)
    # adulterado: uma latência alterada, hash do índice não muda
    alterada = _tabela_sintetica(semente=0)
    alterada.loc[5, "latencia_conjunto_ms"] += 1.0
    _gravar_bruto(raw, "N005_C04_xi0.50", 0, alterada)
    try:
        carregar_execucao(lida[0], raw)
    except ValueError as e:
        assert "SHA-256" in str(e)
    else:
        raise AssertionError("arquivo adulterado deveria falhar")
    # a outra execução continua carregando
    assert len(carregar_execucao(lida[1], raw)) == 200
    # ausente
    (raw / nome_dados("N005_C04_xi0.50", 1)).unlink()
    try:
        carregar_execucao(lida[1], raw)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("arquivo ausente deveria falhar")
    # número de linhas diferente do índice (hash confere, linhas não)
    base2, raw2, indice2, linhas2 = _preparar()
    l2 = ler_indice(indice2)[0]
    l2["n_requisicoes"] = 199
    try:
        carregar_execucao(l2, raw2)
    except ValueError as e:
        assert "linhas" in str(e)
    else:
        raise AssertionError("contagem de linhas divergente deveria falhar")
    # sem verificação de hash, o adulterado carrega (uso só em testes)
    assert len(carregar_execucao(lida[0], raw, verificar_hash=False)) == 200


def teste_limiares():
    base, raw, indice, linhas = _preparar()
    limiares = carregar_limiares(_limiares_json(base))
    assert limiares_do_xi(limiares, 0.5) == {"p90": 18.0, "p95": 22.0, "p99": 36.0}
    assert limiares_do_xi(limiares, 0.2)["p99"] == 33.0
    try:
        limiares_do_xi(limiares, 0.8)
    except ValueError:
        pass
    else:
        raise AssertionError("xi ausente deveria falhar")
    ruim = base / "ruim.json"
    ruim.write_text(json.dumps({"regra": "x", "percentis": {"p90": 0.9}, "limiares": []}))
    try:
        carregar_limiares(ruim)
    except ValueError:
        pass
    else:
        raise AssertionError("percentis inesperados deveriam falhar")


def main() -> None:
    testes = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"ok  {t.__name__}")
    print("Carga funcionando corretamente.")


if __name__ == "__main__":
    main()
