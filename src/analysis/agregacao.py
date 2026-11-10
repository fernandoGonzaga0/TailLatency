"""Agregação por cenário com IC de 95% entre replicações — Etapa 8 (Fase 6, item 1).

Para cada execução (cenário x replicação) calcula as métricas de
`metricas.metricas_execucao` sobre o dado **bruto**, aplicando os limiares de
`limiares.json`; depois agrupa as replicações de cada cenário e dá, por métrica,
a média e a semilargura do IC de 95%.

IC (decisão de 10/11/2026)
--------------------------
IC t de Student entre as n replicações independentes do cenário (n − 1 graus de
liberdade; com n = 5, t = 2,776): média ± t · s / √n, com s o desvio-padrão
amostral (ddof = 1). Vale para todos os cenários, com ou sem contenção. Não é o
IC binomial: com contenção as requisições de um laço são correlacionadas, então o
IC tem de vir da variação entre replicações independentes. O IC de Wilson
continua só na validação sem contenção (`src/simulation/validacao.py`).
As médias de razões (p99/p50) são médias das razões das replicações.

Saídas
------
- `data/metricas_cenarios.csv` (120 linhas, **versionado**): id, N, C, ξ,
  `sem_contencao`, `n_replicacoes` e, por métrica, `<m>_media` e `<m>_ic95`
  (semilargura). É a fonte dos números do texto.
- `results/metricas_execucoes.csv` (600 linhas, ignorada pelo git): as métricas
  de cada execução, usadas na conferência e nos gráficos.

Gravação determinística (floats com `repr`, `\\n`, escrita atômica): os mesmos
dados geram o mesmo arquivo, byte a byte.

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.agregacao --gerar
    python -m src.analysis.teste_agregacao
"""

import argparse
import csv
import io
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

from src.analysis.carga import (
    CAMINHO_INDICE,
    CAMINHO_LIMIARES,
    PASTA_RAW,
    RAIZ,
    carregar_execucao,
    carregar_limiares,
    ler_indice,
    limiares_do_xi,
)
from src.analysis.metricas import METRICAS, metricas_execucao

CAMINHO_CENARIOS = RAIZ / "data" / "metricas_cenarios.csv"
CAMINHO_EXECUCOES = RAIZ / "results" / "metricas_execucoes.csv"
CAMINHO_GRADE = RAIZ / "config" / "grade.toml"
NIVEL = 0.95

IDENTIFICACAO = ("id", "n_fan_out", "concorrencia", "xi", "replicacao")
COLUNAS_EXECUCOES = (*IDENTIFICACAO, "n_requisicoes", *METRICAS)
COLUNAS_CENARIOS = (
    "id", "n_fan_out", "concorrencia", "xi", "sem_contencao", "n_replicacoes",
    *(f"{m}{sufixo}" for m in METRICAS for sufixo in ("_media", "_ic95")),
)


def ic95_t(valores) -> tuple[float, float]:
    """(média, semilargura do IC t de 95%) de valores independentes (n >= 2)."""
    x = np.asarray(valores, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("o IC entre replicações exige pelo menos 2 replicações")
    media = float(np.mean(x))
    desvio = float(np.std(x, ddof=1))
    t = float(stats.t.ppf(0.5 + NIVEL / 2, n - 1))
    return media, t * desvio / math.sqrt(n)


def capacidade_da_grade(caminho: Path | str = CAMINHO_GRADE) -> int:
    """Lê `capacidade` (atendentes por API) de config/grade.toml."""
    import tomllib

    with open(caminho, "rb") as f:
        return int(tomllib.load(f)["geral"]["capacidade"])


def tabela_execucoes(
    linhas_indice: list[dict],
    pasta_raw: Path | str = PASTA_RAW,
    limiares: dict | None = None,
    verificar_hash: bool = True,
    progresso=None,
) -> list[dict]:
    """Métricas de cada execução do índice, recalculadas sobre o bruto."""
    limiares = limiares if limiares is not None else carregar_limiares()
    resultado = []
    inicio = time.monotonic()
    total = len(linhas_indice)
    for i, linha in enumerate(linhas_indice, start=1):
        tabela = carregar_execucao(linha, pasta_raw, verificar_hash=verificar_hash)
        m = metricas_execucao(tabela, limiares_do_xi(limiares, linha["xi"]))
        resultado.append({
            **{c: linha[c] for c in IDENTIFICACAO},
            **{c: m[c] for c in ("n_requisicoes", *METRICAS)},
        })
        if progresso is not None and (i % 100 == 0 or i == total):
            progresso(f"  {i}/{total} execuções ({time.monotonic() - inicio:.0f} s)")
    resultado.sort(key=lambda x: (x["id"], x["replicacao"]))
    return resultado


def agregar_cenarios(execucoes: list[dict], capacidade: int) -> list[dict]:
    """Uma linha por cenário: média e IC de 95% entre as replicações de cada métrica.

    `sem_contencao` é verdadeiro quando C <= capacidade (cada API recebe no máximo
    `capacidade` chamadas simultâneas).
    """
    grupos: dict[str, list[dict]] = {}
    for x in execucoes:
        grupos.setdefault(x["id"], []).append(x)
    linhas = []
    for id_cenario in sorted(grupos):
        g = sorted(grupos[id_cenario], key=lambda x: x["replicacao"])
        primeiro = g[0]
        for x in g:
            if (x["n_fan_out"], x["concorrencia"], x["xi"]) != (
                primeiro["n_fan_out"], primeiro["concorrencia"], primeiro["xi"],
            ):
                raise ValueError(f"{id_cenario}: execuções com parâmetros diferentes")
        linha = {
            "id": id_cenario,
            "n_fan_out": primeiro["n_fan_out"],
            "concorrencia": primeiro["concorrencia"],
            "xi": primeiro["xi"],
            "sem_contencao": primeiro["concorrencia"] <= capacidade,
            "n_replicacoes": len(g),
        }
        for m in METRICAS:
            media, semilargura = ic95_t([x[m] for x in g])
            linha[f"{m}_media"] = media
            linha[f"{m}_ic95"] = semilargura
        linhas.append(linha)
    return linhas


def _texto(valor) -> str:
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, float):
        return repr(valor)
    return "" if valor is None else str(valor)


def conteudo_csv(linhas: list[dict], colunas) -> bytes:
    saida = io.StringIO(newline="")
    escritor = csv.writer(saida, lineterminator="\n")
    escritor.writerow(colunas)
    for linha in linhas:
        escritor.writerow([_texto(linha[c]) for c in colunas])
    return saida.getvalue().encode("utf-8")


def gravar_csv(linhas: list[dict], colunas, caminho: Path | str) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    temporario.write_bytes(conteudo_csv(linhas, colunas))
    temporario.replace(caminho)
    return caminho


def main() -> None:
    ap = argparse.ArgumentParser(description="Métricas por execução e por cenário (Fase 6)")
    ap.add_argument("--gerar", action="store_true", help="calcula e grava as duas tabelas")
    ap.add_argument("--indice", default=str(CAMINHO_INDICE))
    ap.add_argument("--raw", default=str(PASTA_RAW))
    ap.add_argument("--limiares", default=str(CAMINHO_LIMIARES))
    ap.add_argument("--saida-cenarios", default=str(CAMINHO_CENARIOS))
    ap.add_argument("--saida-execucoes", default=str(CAMINHO_EXECUCOES))
    ap.add_argument("--capacidade", type=int, default=None,
                    help="atendentes por API (padrão: lê config/grade.toml)")
    args = ap.parse_args()
    if not args.gerar:
        ap.error("use --gerar")

    capacidade = args.capacidade if args.capacidade is not None else capacidade_da_grade()
    linhas_indice = ler_indice(args.indice)
    print(f"{len(linhas_indice)} execuções no índice; capacidade = {capacidade}")
    execucoes = tabela_execucoes(
        linhas_indice, args.raw, carregar_limiares(args.limiares), progresso=print
    )
    cenarios = agregar_cenarios(execucoes, capacidade)
    gravar_csv(execucoes, COLUNAS_EXECUCOES, args.saida_execucoes)
    gravar_csv(cenarios, COLUNAS_CENARIOS, args.saida_cenarios)
    n_sem = sum(1 for c in cenarios if c["sem_contencao"])
    print(f"{len(cenarios)} cenários ({n_sem} sem contenção, {len(cenarios) - n_sem} com)")
    print(f"gravado: {args.saida_execucoes}")
    print(f"gravado: {args.saida_cenarios}")


if __name__ == "__main__":
    main()
