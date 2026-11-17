"""Concorrência efetiva — Etapa 8 (Fase 6, item 9). Descritivo; precisa do dado bruto.

Pergunta: a cauda pesada "prende" clientes? Num sistema fechado sem tempo de
pensamento, os C laços estão sempre dentro de uma requisição. Se uma requisição
fica presa atrás de uma chamada gigante, as outras N−1 chamadas dela já terminaram
e liberaram os servidores: aquele laço deixa de gerar demanda. Com ξ alto isso
reduziria a concorrência efetiva e explicaria por que o excesso sobre a teoria
(item 7) diminui com ξ. Este módulo mede isso.

Definição (fixada em 14/11/2026, antes de rodar):

- requisição presa: latencia_conjunto_ms > K x limiar_p99 do ξ do cenário, K = 10
  (estritamente maior);
- T = (maior fim_ms) − (menor inicio_ms) entre as requisições medidas;
- f_preso = soma das durações das requisições presas, recortadas em [menor inicio,
  maior fim], dividida por C x T (fração do tempo-laço gasto em requisições presas);
- c_efetivo = C x (1 − f_preso);
- frac_req_presas = fração das requisições medidas que são presas.

Aproximação: requisições iniciadas e ainda não concluídas no fim da execução não
estão no arquivo, e as do aquecimento foram descartadas; o recorte em
[menor inicio, maior fim] trata essas bordas de forma simples. A medida serve para
comparar ξ e C entre cenários, não como valor exato.

Agregação entre as replicações por IC t de 95% (como no item 1). Saída versionada:
data/concorrencia_efetiva.csv (120 linhas). Lê o bruto: roda só no computador
principal (data/raw não está no Git).

    python -m src.analysis.concorrencia_efetiva --gerar
    python -m src.analysis.teste_concorrencia_efetiva
"""

import argparse
import time

import numpy as np

from src.analysis.agregacao import capacidade_da_grade, gravar_csv, ic95_t
from src.analysis.carga import (
    PASTA_RAW,
    RAIZ,
    carregar_execucao,
    carregar_limiares,
    ler_indice,
    limiares_do_xi,
)

K_PRESA = 10.0
CAMINHO_SAIDA = RAIZ / "data" / "concorrencia_efetiva.csv"
MEDIDAS = ("f_preso", "c_efetivo", "frac_req_presas", "tempo_total_ms")
COLUNAS = (
    "id", "n_fan_out", "concorrencia", "xi", "sem_contencao", "n_replicacoes",
    *(f"{m}_{s}" for m in MEDIDAS for s in ("media", "ic95")),
)


def medidas_execucao(tabela, concorrencia: int, limiar_p99_ms: float, k: float = K_PRESA) -> dict:
    """f_preso, c_efetivo, frac_req_presas e tempo_total_ms de uma execução."""
    if concorrencia < 1 or limiar_p99_ms <= 0 or k <= 0:
        raise ValueError("concorrencia >= 1, limiar > 0 e k > 0")
    ini = tabela["inicio_ms"].to_numpy(dtype=float)
    fim = tabela["fim_ms"].to_numpy(dtype=float)
    lat = tabela["latencia_conjunto_ms"].to_numpy(dtype=float)
    if len(ini) == 0:
        raise ValueError("execução sem requisições")
    t0, t1 = float(ini.min()), float(fim.max())
    total = t1 - t0
    if total <= 0:
        raise ValueError("duração total não positiva")
    presa = lat > k * limiar_p99_ms
    dur = np.clip(fim, t0, t1) - np.clip(ini, t0, t1)
    f = float(dur[presa].sum()) / (concorrencia * total)
    return {
        "f_preso": f,
        "c_efetivo": concorrencia * (1.0 - f),
        "frac_req_presas": float(presa.mean()),
        "tempo_total_ms": total,
    }


def agregar(execucoes: list[dict], capacidade: int) -> list[dict]:
    """Uma linha por cenário: média e IC t de 95% de cada medida entre as replicações."""
    grupos: dict[str, list[dict]] = {}
    for e in execucoes:
        grupos.setdefault(e["id"], []).append(e)
    saida = []
    for id_, g in sorted(grupos.items()):
        base = g[0]
        linha = {
            "id": id_, "n_fan_out": base["n_fan_out"], "concorrencia": base["concorrencia"],
            "xi": base["xi"], "sem_contencao": base["concorrencia"] <= capacidade,
            "n_replicacoes": len(g),
        }
        for m in MEDIDAS:
            media, semi = ic95_t([x[m] for x in g])
            linha[f"{m}_media"], linha[f"{m}_ic95"] = media, semi
        saida.append(linha)
    return saida


def calcular(linhas_indice, pasta_raw=PASTA_RAW, limiares=None, verificar_hash=True, progresso=None):
    limiares = limiares if limiares is not None else carregar_limiares()
    inicio, total, saida = time.monotonic(), len(linhas_indice), []
    for i, linha in enumerate(linhas_indice, start=1):
        tabela = carregar_execucao(linha, pasta_raw, verificar_hash=verificar_hash)
        lim99 = limiares_do_xi(limiares, linha["xi"])["p99"]
        m = medidas_execucao(tabela, linha["concorrencia"], lim99)
        saida.append({"id": linha["id"], "n_fan_out": linha["n_fan_out"],
                      "concorrencia": linha["concorrencia"], "xi": linha["xi"],
                      "replicacao": linha["replicacao"], **m})
        if progresso is not None and (i % 100 == 0 or i == total):
            progresso(f"  {i}/{total} execuções ({time.monotonic() - inicio:.0f} s)")
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description="Concorrência efetiva (Fase 6, item 9)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--saida", default=str(CAMINHO_SAIDA))
    ap.add_argument("--ids", nargs="*", default=None, help="restringe a estes cenários (teste rápido)")
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    linhas = ler_indice()
    if a.ids:
        linhas = [x for x in linhas if x["id"] in set(a.ids)]
    execs = calcular(linhas, progresso=print)
    tabela = agregar(execs, capacidade_da_grade())
    gravar_csv(tabela, COLUNAS, a.saida)
    print(f"\n{'cenário':18s} {'C':>3s} {'f_preso':>9s} {'c_efetivo':>10s} {'req presas':>11s}")
    for x in tabela:
        if not x["sem_contencao"] or x["concorrencia"] == 4:
            print(f"{x['id']:18s} {x['concorrencia']:3d} {x['f_preso_media']:9.4f} "
                  f"{x['c_efetivo_media']:10.2f} {x['frac_req_presas_media']:11.5f}")
    print(f"\ngravado: {a.saida}")


if __name__ == "__main__":
    main()
