"""Robustez ao transiente — Fase 6, verificação A (fim de execução e aquecimento).

Duas perguntas sobre a simulação em sistema fechado (requisicoes.py):

A1. Fim de execução. Os C laços param de disparar quando o contador chega ao total;
    as últimas requisições competem com menos laços, então a concorrência decai. A
    violação das últimas 500 requisições (por id, ordem de disparo) difere da das
    demais? Usa só o dado bruto já gerado (sem simular de novo).
A2. Aquecimento. O aquecimento é por contagem (500 requisições), não por tempo. A
    violação medida com 500 de aquecimento difere da medida com 2.000? Reexecuta
    poucos cenários (mesmas seeds, mesmo número de requisições medidas), com
    aquecimentos de 0, 500 (o atual) e 2.000.

Critérios (fixados em 17/11/2026, antes de rodar; violação = latencia_conjunto_ms
estritamente acima do limiar p95 do ξ, como em todo o trabalho):

- A1: nos cenários com C >= 8 e ξ >= 0,8 (30 cenários), a violação das últimas
  500 requisições não difere da das demais em pelo menos 90% dos cenários;
- A2: nos cenários simulados, a violação com aquecimento 500 não difere da com 2.000
  em pelo menos 90% dos cenários. Aquecimento 0 é só descritivo.

"Não difere" = pelo critério de sobreposição de IC usado em toda a Fase 6: as médias
diferem por no máximo a soma das semilarguras dos IC t de 95% (10 replicações).
Como as duas medidas vêm da mesma execução (A1) ou da mesma seed (A2), elas são
positivamente correlacionadas e a sobreposição é indulgente; por isso também se
reporta, **sem valer como critério**, o IC t da diferença pareada por replicação
(mais rigoroso). Se os dois discordarem, a discordância é relatada.

Saídas versionadas: data/robustez_fim_execucao.csv, data/robustez_aquecimento.csv e
data/robustez_veredito.csv. Lê o bruto: roda só no computador principal.

    python -m src.analysis.robustez_transiente --gerar --processos 4
    python -m src.analysis.teste_robustez_transiente
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.analysis.agregacao import gravar_csv, ic95_t
from src.analysis.carga import (
    PASTA_RAW,
    RAIZ,
    carregar_execucao,
    carregar_limiares,
    ler_indice,
    limiares_do_xi,
)
from src.simulation.cenarios import buscar_cenario, carregar_grade
from src.simulation.requisicoes import rodar_cenario

JANELA_FIM = 500
CS_ALTOS = (8, 10, 12)
XIS_ALTOS = (0.8, 1.1)
NS_AQUECIMENTO = (10, 50)
CS_AQUECIMENTO = (8, 12)
XIS_AQUECIMENTO = (0.8, 1.1)
AQUECIMENTOS = (0, 500, 2000)
MINIMO = 0.90

CAMINHO_FIM = RAIZ / "data" / "robustez_fim_execucao.csv"
CAMINHO_AQUECIMENTO = RAIZ / "data" / "robustez_aquecimento.csv"
CAMINHO_VEREDITO = RAIZ / "data" / "robustez_veredito.csv"

COLUNAS_FIM = (
    "id", "n_fan_out", "concorrencia", "xi", "n_replicacoes",
    "viol_fim_media", "viol_fim_ic95", "viol_resto_media", "viol_resto_ic95",
    "dif_media", "dif_ic95", "difere_sobreposicao", "difere_pareada",
)
COLUNAS_AQUECIMENTO = (
    "id", "n_fan_out", "concorrencia", "xi", "n_replicacoes",
    *(f"viol_w{w}_{s}" for w in AQUECIMENTOS for s in ("media", "ic95")),
    "dif_2000_500_media", "dif_2000_500_ic95",
    "difere_sobreposicao_2000_500", "difere_pareada_2000_500",
    "dif_500_0_media", "dif_500_0_ic95",
    "difere_sobreposicao_500_0", "difere_pareada_500_0",
)
COLUNAS_VEREDITO = ("criterio", "tipo", "testes", "passam", "fracao", "minimo", "aprovado")


# ---------------------------------------------------------------- comparações

def difere_sobreposicao(m1: float, i1: float, m2: float, i2: float) -> bool:
    """Diferem se a distância entre as médias passa a soma das semilarguras dos IC."""
    return abs(m1 - m2) > i1 + i2


def violacao(lat, limiar_ms: float) -> float:
    """Fração de latências estritamente acima do limiar."""
    return float(np.mean(np.asarray(lat, dtype=float) > limiar_ms))


# ------------------------------------------------------------------------ A1

def ids_fim(cenarios) -> list[str]:
    return sorted(c.id for c in cenarios if c.concorrencia in CS_ALTOS and c.xi in XIS_ALTOS)


def fim_execucao_execucao(tabela, limiar_p95_ms: float, janela: int = JANELA_FIM) -> dict:
    """Violação das últimas `janela` requisições (por id) e das demais, de uma execução."""
    if janela < 1:
        raise ValueError("janela >= 1")
    t = tabela.sort_values("id_requisicao")
    lat = t["latencia_conjunto_ms"].to_numpy(dtype=float)
    if len(lat) <= janela:
        raise ValueError("a execução precisa ter mais requisições que a janela")
    return {
        "viol_fim": violacao(lat[-janela:], limiar_p95_ms),
        "viol_resto": violacao(lat[:-janela], limiar_p95_ms),
        "viol_total": violacao(lat, limiar_p95_ms),
        "n_total": len(lat),
    }


def agregar_fim(execucoes: list[dict]) -> list[dict]:
    grupos: dict[str, list[dict]] = {}
    for e in execucoes:
        grupos.setdefault(e["id"], []).append(e)
    saida = []
    for id_, g in sorted(grupos.items()):
        b = g[0]
        mf, i_f = ic95_t([x["viol_fim"] for x in g])
        mr, i_r = ic95_t([x["viol_resto"] for x in g])
        md, i_d = ic95_t([x["viol_fim"] - x["viol_resto"] for x in g])
        saida.append({
            "id": id_, "n_fan_out": b["n_fan_out"], "concorrencia": b["concorrencia"],
            "xi": b["xi"], "n_replicacoes": len(g),
            "viol_fim_media": mf, "viol_fim_ic95": i_f,
            "viol_resto_media": mr, "viol_resto_ic95": i_r,
            "dif_media": md, "dif_ic95": i_d,
            "difere_sobreposicao": difere_sobreposicao(mf, i_f, mr, i_r),
            "difere_pareada": abs(md) > i_d,
        })
    return saida


def calcular_fim(linhas_indice, pasta_raw=PASTA_RAW, limiares=None, verificar_hash=True, progresso=None):
    limiares = limiares if limiares is not None else carregar_limiares()
    inicio, total, saida = time.monotonic(), len(linhas_indice), []
    for i, linha in enumerate(linhas_indice, start=1):
        tabela = carregar_execucao(linha, pasta_raw, verificar_hash=verificar_hash)
        lim95 = limiares_do_xi(limiares, linha["xi"])["p95"]
        m = fim_execucao_execucao(tabela, lim95)
        saida.append({"id": linha["id"], "n_fan_out": linha["n_fan_out"],
                      "concorrencia": linha["concorrencia"], "xi": linha["xi"],
                      "replicacao": linha["replicacao"], **m})
        if progresso is not None and (i % 50 == 0 or i == total):
            progresso(f"  A1: {i}/{total} execuções ({time.monotonic() - inicio:.0f} s)")
    return saida


# ------------------------------------------------------------------------ A2

def ids_aquecimento(cenarios) -> list[str]:
    return sorted(
        c.id for c in cenarios
        if c.n_fan_out in NS_AQUECIMENTO and c.concorrencia in CS_AQUECIMENTO and c.xi in XIS_AQUECIMENTO
    )


def violacao_com_aquecimento(cenario, replicacao: int, aquecimento: int, limiar_p95_ms: float) -> dict:
    """Reexecuta a réplica (mesma seed e mesmo nº de requisições medidas) com outro aquecimento."""
    res = rodar_cenario(
        n_fan_out=cenario.n_fan_out, concorrencia=cenario.concorrencia,
        n_requisicoes=cenario.n_requisicoes, params=cenario.params,
        capacidade=cenario.capacidade, seed=cenario.seed(replicacao),
        n_aquecimento=aquecimento,
    )
    lat = [r.latencia_conjunto for r in res]
    return {"viol": violacao(lat, limiar_p95_ms), "n": len(lat)}


def _tarefa(args):
    id_, replicacao, aquecimento, limiar = args
    cenario = buscar_cenario(carregar_grade(), id_)
    r = violacao_com_aquecimento(cenario, replicacao, aquecimento, limiar)
    return {"id": id_, "n_fan_out": cenario.n_fan_out, "concorrencia": cenario.concorrencia,
            "xi": cenario.xi, "replicacao": replicacao, "aquecimento": aquecimento, **r}


def _ler_cache(cache) -> dict:
    feitos = {}
    if cache is not None and Path(cache).exists():
        with open(cache, encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    r = json.loads(linha)
                    feitos[(r["id"], r["replicacao"], r["aquecimento"])] = r
    return feitos


def calcular_aquecimento(ids, n_replicacoes, limiares=None, processos=1, progresso=None, cache=None):
    """Reexecuta os cenários com aquecimento 0, 500 e 2.000.

    `cache` (opcional): arquivo JSON por linha com as simulações já concluídas; permite
    interromper e retomar sem refazer (cada resultado é determinístico pela seed).
    """
    limiares = limiares if limiares is not None else carregar_limiares()
    grade = {c.id: c for c in carregar_grade()}
    tarefas = []
    for id_ in ids:
        lim = limiares_do_xi(limiares, grade[id_].xi)["p95"]
        for rep in range(n_replicacoes):
            for w in AQUECIMENTOS:
                tarefas.append((id_, rep, w, lim))
    tarefas.sort(key=lambda t: -grade[t[0]].n_fan_out)  # as mais pesadas primeiro
    feitos = _ler_cache(cache)
    pendentes = [t for t in tarefas if (t[0], t[1], t[2]) not in feitos]
    inicio, total = time.monotonic(), len(tarefas)
    arq = open(cache, "a", encoding="utf-8") if cache is not None else None

    def guardar(r):
        feitos[(r["id"], r["replicacao"], r["aquecimento"])] = r
        if arq is not None:
            arq.write(json.dumps(r) + "\n")
            arq.flush()
        if progresso and (len(feitos) % 20 == 0 or len(feitos) == total):
            progresso(f"  A2: {len(feitos)}/{total} simulações ({time.monotonic() - inicio:.0f} s)")

    try:
        if processos <= 1:
            for t in pendentes:
                guardar(_tarefa(t))
        else:
            with ProcessPoolExecutor(max_workers=processos) as ex:
                futuros = [ex.submit(_tarefa, t) for t in pendentes]
                for fut in as_completed(futuros):
                    guardar(fut.result())
    finally:
        if arq is not None:
            arq.close()
    return [feitos[(t[0], t[1], t[2])] for t in tarefas]


def conferir_reproducao(exec_aquecimento: list[dict], exec_fim: list[dict], n_requisicoes: int) -> int:
    """Com aquecimento 500 a simulação tem de reproduzir o bruto (mesma seed): confere.

    Devolve quantas réplicas foram conferidas; falha se alguma divergir por mais de
    uma requisição (tolerância para diferença de versão do SciPy na última casa).
    """
    bruto = {(e["id"], e["replicacao"]): e["viol_total"] for e in exec_fim}
    n = 0
    for e in exec_aquecimento:
        if e["aquecimento"] != 500:
            continue
        ref = bruto.get((e["id"], e["replicacao"]))
        if ref is None:
            continue
        if abs(e["viol"] - ref) > 1.0 / n_requisicoes + 1e-12:
            raise ValueError(
                f"{e['id']} rep{e['replicacao']}: a reexecução com aquecimento 500 "
                f"({e['viol']:.6f}) não reproduz o bruto ({ref:.6f})"
            )
        n += 1
    return n


def agregar_aquecimento(execucoes: list[dict]) -> list[dict]:
    grupos: dict[str, list[dict]] = {}
    for e in execucoes:
        grupos.setdefault(e["id"], []).append(e)
    saida = []
    for id_, g in sorted(grupos.items()):
        b = g[0]
        por_w = {w: {x["replicacao"]: x["viol"] for x in g if x["aquecimento"] == w} for w in AQUECIMENTOS}
        reps = sorted(por_w[500])
        for w in AQUECIMENTOS:
            if sorted(por_w[w]) != reps:
                raise ValueError(f"{id_}: réplicas diferentes entre os aquecimentos")
        linha = {"id": id_, "n_fan_out": b["n_fan_out"], "concorrencia": b["concorrencia"],
                 "xi": b["xi"], "n_replicacoes": len(reps)}
        est = {}
        for w in AQUECIMENTOS:
            est[w] = ic95_t([por_w[w][r] for r in reps])
            linha[f"viol_w{w}_media"], linha[f"viol_w{w}_ic95"] = est[w]
        for rotulo, a, b_ in (("2000_500", 2000, 500), ("500_0", 500, 0)):
            md, i_d = ic95_t([por_w[a][r] - por_w[b_][r] for r in reps])
            linha[f"dif_{rotulo}_media"], linha[f"dif_{rotulo}_ic95"] = md, i_d
            linha[f"difere_sobreposicao_{rotulo}"] = difere_sobreposicao(*est[a], *est[b_])
            linha[f"difere_pareada_{rotulo}"] = abs(md) > i_d
        saida.append(linha)
    return saida


# ------------------------------------------------------------------- veredito

def _linha_veredito(criterio, tipo, flags_difere):
    testes = len(flags_difere)
    passam = sum(1 for d in flags_difere if not d)
    fracao = passam / testes if testes else 0.0
    return {"criterio": criterio, "tipo": tipo, "testes": testes, "passam": passam,
            "fracao": fracao, "minimo": MINIMO, "aprovado": fracao >= MINIMO - 1e-12}


def veredito(fim: list[dict], aquecimento: list[dict]) -> list[dict]:
    return [
        _linha_veredito("A1_fim_execucao_sobreposicao", "criterio", [x["difere_sobreposicao"] for x in fim]),
        _linha_veredito("A1_fim_execucao_pareada", "descritivo", [x["difere_pareada"] for x in fim]),
        _linha_veredito("A2_aquecimento_2000_vs_500_sobreposicao", "criterio",
                        [x["difere_sobreposicao_2000_500"] for x in aquecimento]),
        _linha_veredito("A2_aquecimento_2000_vs_500_pareada", "descritivo",
                        [x["difere_pareada_2000_500"] for x in aquecimento]),
        _linha_veredito("A2_aquecimento_500_vs_0_sobreposicao", "descritivo",
                        [x["difere_sobreposicao_500_0"] for x in aquecimento]),
        _linha_veredito("A2_aquecimento_500_vs_0_pareada", "descritivo",
                        [x["difere_pareada_500_0"] for x in aquecimento]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Robustez ao transiente (Fase 6, verificação A)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--processos", type=int, default=1)
    ap.add_argument("--so-fim", action="store_true", help="só A1 (não simula)")
    ap.add_argument("--cache", default=None, help="arquivo para guardar/retomar as simulações do A2")
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    grade = carregar_grade()
    n_rep = grade[0].n_replicacoes
    n_req = grade[0].n_requisicoes
    indice = ler_indice()

    ids1 = set(ids_fim(grade))
    execs_fim = calcular_fim([x for x in indice if x["id"] in ids1], progresso=print)
    tab_fim = agregar_fim(execs_fim)
    gravar_csv(tab_fim, COLUNAS_FIM, CAMINHO_FIM)
    print(f"\nA1: {len(tab_fim)} cenários -> {CAMINHO_FIM}")
    if a.so_fim:
        return 0

    execs_aq = calcular_aquecimento(ids_aquecimento(grade), n_rep, processos=a.processos,
                                    progresso=print, cache=a.cache)
    conferidas = conferir_reproducao(execs_aq, execs_fim, n_req)
    print(f"  reprodução do bruto com aquecimento 500 conferida em {conferidas} réplicas")
    tab_aq = agregar_aquecimento(execs_aq)
    gravar_csv(tab_aq, COLUNAS_AQUECIMENTO, CAMINHO_AQUECIMENTO)
    ver = veredito(tab_fim, tab_aq)
    gravar_csv(ver, COLUNAS_VEREDITO, CAMINHO_VEREDITO)

    print(f"\n{'critério':44s} {'tipo':10s} {'passam':>8s} {'fração':>7s}  aprovado")
    for v in ver:
        print(f"{v['criterio']:44s} {v['tipo']:10s} {v['passam']:3d}/{v['testes']:<4d} "
              f"{v['fracao']:7.3f}  {'sim' if v['aprovado'] else 'NÃO'}")
    print(f"\ngravados: {CAMINHO_AQUECIMENTO.name}, {CAMINHO_VEREDITO.name}")
    return 0 if all(v["aprovado"] for v in ver if v["tipo"] == "criterio") else 1


if __name__ == "__main__":
    raise SystemExit(main())
