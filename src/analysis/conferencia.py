"""Conferência cruzada das marcações por limiar — Etapa 8 (Fase 6, item 2).

A Fase 5 gravou a marcação por limiar (`viola_p90/p95/p99`) em `data/processed/`;
a Fase 6 reaplica os limiares sobre o dado **bruto** (que continua sem limiar).
Este módulo confere que os dois caminhos concordam. Para cada execução do índice:

1. recalcula `latencia_conjunto_ms > limiar` sobre o bruto, com os limiares de
   `limiares.json` (regra estrita: igual ao limiar não é violação);
2. compara, requisição por requisição, com as colunas `viola_*` do arquivo marcado
   (e confere que a latência do marcado é a do bruto);
3. compara as contagens e as frações com o registro `marcacao_<id>_rep<k>.json`,
   os limiares do registro com `limiares.json`, e os SHA-256 do bruto e do marcado
   com os do registro e do índice;
4. compara com o índice as frações de violação, o p50, o p99, o p99/p50, a
   latência média e a espera média, e os limiares gravados.

Também confere a cobertura: cada execução do índice tem bruto, manifesto, marcado
e registro, e não há arquivo de execução sobrando em `data/raw/` ou
`data/processed/`.

Qualquer divergência é **erro de código ou de dado, não resultado**. A conferência
não altera nenhum arquivo.

Tolerâncias
-----------
- Contagens de violação e valores 0/1 do marcado: igualdade exata (são inteiros).
- Frações de violação do índice: 1e-12 (são contagem / n).
- Demais números (percentis, médias, limiares): 1e-9 relativo. Vêm de cálculos em
  ponto flutuante que podem diferir na casa de 1e-13 entre versões de NumPy, sem
  erro nenhum (a análise feita na VM de testes diferiu do índice em até 5e-13).
- Hashes: igualdade exata (identificam os arquivos deste computador).

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.conferencia --tudo
    python -m src.analysis.conferencia --id N010_C06_xi0.50
    python -m src.analysis.teste_conferencia

Código de saída 0 se tudo confere; 1 se houver divergência.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np

from src.analysis.carga import (
    CAMINHO_INDICE,
    CAMINHO_LIMIARES,
    PASTA_PROCESSED,
    PASTA_RAW,
    carregar_execucao,
    carregar_limiares,
    ler_indice,
    limiares_do_xi,
)
from src.analysis.metricas import NOMES_LIMIAR, contagens_violacao, metricas_execucao
from src.simulation.saida import ler_requisicoes, nome_dados, sha256_conteudo

TOL_REL = 1e-9
TOL_FRACAO = 1e-12
COLUNAS_VIOLACAO = tuple(f"viola_{n}" for n in NOMES_LIMIAR)


def nome_manifesto(id_cenario: str, replicacao: int) -> str:
    return f"manifesto_{id_cenario}_rep{replicacao}.json"


def nome_registro(id_cenario: str, replicacao: int) -> str:
    return f"marcacao_{id_cenario}_rep{replicacao}.json"


def _igual(a: float, b: float, tol: float = TOL_REL) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=1e-12)


def _carregar_json(caminho: Path) -> dict:
    import json

    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def conferir_execucao(
    linha: dict,
    limiares: dict,
    pasta_raw: Path | str = PASTA_RAW,
    pasta_processed: Path | str = PASTA_PROCESSED,
) -> list[str]:
    """Divergências de uma execução (lista vazia = confere).

    `linha` é uma linha de `carga.ler_indice`; `limiares` vem de `carregar_limiares`.
    """
    pasta_raw, pasta_processed = Path(pasta_raw), Path(pasta_processed)
    id_, rep = linha["id"], linha["replicacao"]
    problemas: list[str] = []

    # --- bruto (a carga já confere SHA-256 e número de linhas contra o índice) ---
    try:
        bruto = carregar_execucao(linha, pasta_raw)
    except (FileNotFoundError, ValueError) as erro:
        return [f"bruto: {erro}"]
    lim = limiares_do_xi(limiares, linha["xi"])
    lat = bruto["latencia_conjunto_ms"].to_numpy(dtype=float)
    n = len(lat)
    recalculado = {nome: lat > lim[nome] for nome in NOMES_LIMIAR}
    contagens = contagens_violacao(lat, lim)

    # --- limiares do índice contra limiares.json ---
    for nome in NOMES_LIMIAR:
        if not _igual(linha[f"limiar_{nome}_ms"], lim[nome]):
            problemas.append(
                f"índice: limiar_{nome}_ms = {linha[f'limiar_{nome}_ms']!r}, "
                f"limiares.json = {lim[nome]!r}"
            )

    # --- arquivo marcado ---
    caminho_marcado = pasta_processed / nome_dados(id_, rep)
    sha_marcado = None
    if not caminho_marcado.exists():
        problemas.append(f"marcado: arquivo ausente ({caminho_marcado.name})")
    else:
        sha_marcado = sha256_conteudo(caminho_marcado)
        marcado = ler_requisicoes(caminho_marcado)
        faltam = [c for c in COLUNAS_VIOLACAO if c not in marcado]
        if faltam:
            problemas.append(f"marcado: faltam as colunas {faltam}")
        elif len(marcado) != n:
            problemas.append(f"marcado: {len(marcado)} linhas, bruto tem {n}")
        else:
            if not np.array_equal(marcado["latencia_conjunto_ms"].to_numpy(dtype=float), lat):
                problemas.append("marcado: latencia_conjunto_ms difere do bruto")
            if not np.array_equal(marcado["id_requisicao"].to_numpy(), bruto["id_requisicao"].to_numpy()):
                problemas.append("marcado: id_requisicao difere do bruto")
            for nome in NOMES_LIMIAR:
                col = marcado[f"viola_{nome}"].to_numpy()
                if not np.isin(col, (0, 1)).all():
                    problemas.append(f"marcado: viola_{nome} tem valores fora de 0 e 1")
                    continue
                esperado = recalculado[nome].astype(col.dtype)
                if not np.array_equal(col, esperado):
                    dif = int(np.count_nonzero(col != esperado))
                    problemas.append(
                        f"marcado: viola_{nome} difere do recalculado em {dif} requisições "
                        f"(marcado soma {int(col.sum())}, recalculado {contagens[nome]})"
                    )
        if sha_marcado != linha["sha256_marcado"]:
            problemas.append("marcado: SHA-256 do arquivo difere do índice")

    # --- registro de marcação ---
    caminho_registro = pasta_processed / nome_registro(id_, rep)
    if not caminho_registro.exists():
        problemas.append(f"registro: arquivo ausente ({caminho_registro.name})")
    else:
        try:
            reg = _carregar_json(caminho_registro)
            if (reg["id"], reg["replicacao"]) != (id_, rep):
                problemas.append("registro: id ou replicação diferentes da execução")
            if reg["bruto"]["sha256_conteudo"] != linha["sha256_bruto"]:
                problemas.append("registro: SHA-256 do bruto difere do índice")
            if reg["marcado"]["sha256_conteudo"] != linha["sha256_marcado"]:
                problemas.append("registro: SHA-256 do marcado difere do índice")
            if sha_marcado is not None and reg["marcado"]["sha256_conteudo"] != sha_marcado:
                problemas.append("registro: SHA-256 do marcado difere do arquivo marcado")
            if reg["marcado"]["linhas"] != n:
                problemas.append(f"registro: {reg['marcado']['linhas']} linhas, bruto tem {n}")
            for nome in NOMES_LIMIAR:
                if not _igual(reg["limiares_ms"][nome], lim[nome]):
                    problemas.append(
                        f"registro: limiar {nome} = {reg['limiares_ms'][nome]!r}, "
                        f"limiares.json = {lim[nome]!r}"
                    )
                if reg["violacoes"][nome] != contagens[nome]:
                    problemas.append(
                        f"registro: violações {nome} = {reg['violacoes'][nome]}, "
                        f"recalculado {contagens[nome]}"
                    )
                if not _igual(reg["fracao_violacao"][nome], contagens[nome] / n, TOL_FRACAO):
                    problemas.append(f"registro: fração de violação {nome} difere do recalculado")
        except (KeyError, TypeError, ValueError) as erro:
            problemas.append(f"registro: ilegível ({type(erro).__name__}: {erro})")

    # --- índice contra o recalculado ---
    m = metricas_execucao(bruto, lim)
    if linha["n_requisicoes"] != n:
        problemas.append(f"índice: n_requisicoes = {linha['n_requisicoes']}, bruto tem {n}")
    for nome in NOMES_LIMIAR:
        if not _igual(linha[f"violacao_{nome}"], contagens[nome] / n, TOL_FRACAO):
            problemas.append(
                f"índice: violacao_{nome} = {linha[f'violacao_{nome}']!r}, "
                f"recalculado {contagens[nome] / n!r}"
            )
    for coluna in (
        "p50_ms", "p99_ms", "amplificacao_p99_p50", "latencia_media_ms",
        "espera_media_chamada_ms",
    ):
        if not _igual(linha[coluna], m[coluna]):
            problemas.append(f"índice: {coluna} = {linha[coluna]!r}, recalculado {m[coluna]!r}")
    return problemas


def conferir_cobertura(
    linhas: list[dict],
    pasta_raw: Path | str = PASTA_RAW,
    pasta_processed: Path | str = PASTA_PROCESSED,
) -> list[str]:
    """Execuções sem arquivo e arquivos de execução que o índice não conhece."""
    pasta_raw, pasta_processed = Path(pasta_raw), Path(pasta_processed)
    esperados = {
        pasta_raw: set(),
        pasta_processed: set(),
    }
    for x in linhas:
        esperados[pasta_raw] |= {nome_dados(x["id"], x["replicacao"]), nome_manifesto(x["id"], x["replicacao"])}
        esperados[pasta_processed] |= {nome_dados(x["id"], x["replicacao"]), nome_registro(x["id"], x["replicacao"])}
    padroes = ("requisicoes_*.csv.gz", "manifesto_*.json", "marcacao_*.json")
    problemas = []
    for pasta, nomes in esperados.items():
        existentes = {p.name for padrao in padroes for p in pasta.glob(padrao)}
        for nome in sorted(nomes - existentes):
            problemas.append(f"{pasta.name}/: falta {nome}")
        for nome in sorted(existentes - nomes):
            problemas.append(f"{pasta.name}/: {nome} existe e não consta no índice")
    return problemas


def conferir_tudo(
    linhas: list[dict],
    limiares: dict,
    pasta_raw: Path | str = PASTA_RAW,
    pasta_processed: Path | str = PASTA_PROCESSED,
    progresso=None,
) -> dict:
    """Confere todas as execuções e a cobertura.

    Devolve {'execucoes': n, 'divergencias': {(id, rep): [mensagens]}, 'cobertura': [mensagens]}.
    Uma execução com problema não interrompe as outras.
    """
    divergencias: dict = {}
    inicio = time.monotonic()
    total = len(linhas)
    for i, linha in enumerate(linhas, start=1):
        achados = conferir_execucao(linha, limiares, pasta_raw, pasta_processed)
        if achados:
            divergencias[(linha["id"], linha["replicacao"])] = achados
        if progresso is not None and (i % 200 == 0 or i == total):
            progresso(f"  {i}/{total} execuções ({time.monotonic() - inicio:.0f} s)")
    return {
        "execucoes": total,
        "divergencias": divergencias,
        "cobertura": conferir_cobertura(linhas, pasta_raw, pasta_processed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Conferência cruzada das marcações (Fase 6, item 2)")
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--tudo", action="store_true", help="confere todas as execuções do índice")
    modo.add_argument("--id", action="append", metavar="ID", help="só este cenário (repetível)")
    ap.add_argument("--indice", default=str(CAMINHO_INDICE))
    ap.add_argument("--raw", default=str(PASTA_RAW))
    ap.add_argument("--processed", default=str(PASTA_PROCESSED))
    ap.add_argument("--limiares", default=str(CAMINHO_LIMIARES))
    args = ap.parse_args()

    linhas = ler_indice(args.indice)
    if args.id:
        desconhecidos = sorted(set(args.id) - {x["id"] for x in linhas})
        if desconhecidos:
            ap.error(f"cenários fora do índice: {desconhecidos}")
        linhas = [x for x in linhas if x["id"] in args.id]
    limiares = carregar_limiares(args.limiares)
    print(f"conferindo {len(linhas)} execuções (limiares: {Path(args.limiares).name})")
    r = conferir_tudo(linhas, limiares, args.raw, args.processed, progresso=print)

    for (id_, rep), mensagens in sorted(r["divergencias"].items()):
        for msg in mensagens:
            print(f"DIVERGÊNCIA {id_} rep{rep}: {msg}")
    if not args.id:  # a cobertura só faz sentido com o índice inteiro
        for msg in r["cobertura"]:
            print(f"COBERTURA {msg}")
    n_cob = 0 if args.id else len(r["cobertura"])
    ok = r["execucoes"] - len(r["divergencias"])
    print(
        f"{ok} de {r['execucoes']} execuções conferem; {len(r['divergencias'])} com divergência; "
        f"{n_cob} problemas de cobertura"
    )
    tudo_certo = not r["divergencias"] and n_cob == 0
    print("conferência cruzada OK" if tudo_certo else "conferência cruzada NÃO confere")
    raise SystemExit(0 if tudo_certo else 1)


if __name__ == "__main__":
    main()
