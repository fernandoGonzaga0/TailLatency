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
    python -m src.analysis.concorrencia_efetiva --sensibilidade-k
    python -m src.analysis.teste_concorrencia_efetiva

Sensibilidade ao K (Fase 6, verificação B, critério fixado em 17/11/2026 antes de
rodar). K = 10 nunca havia sido variado. `--sensibilidade-k` recalcula as medidas
com K = 5, 10 e 20 (lendo cada bruto uma só vez) e compara a direção das tendências
de f_preso com C (por N e ξ, 20 séries) e com ξ (por N e C, 30 séries): "cresce",
"decresce" ou "nenhuma" (mesma regra de `hipotese.cresce`, aplicada a f_preso e ao
seu oposto). Critério: com K = 5 e com K = 20, a direção coincide com a de K = 10 em
pelo menos 90% das 50 séries. O K = 10 refeito tem de reproduzir
data/concorrencia_efetiva.csv (que não é alterado). Saídas versionadas:
data/concorrencia_efetiva_k.csv e data/concorrencia_efetiva_k_direcao.csv.
"""

import argparse
import csv
import time

import numpy as np

from src.analysis.agregacao import capacidade_da_grade, gravar_csv, ic95_t
from src.analysis.hipotese import cresce
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

KS_SENSIBILIDADE = (5.0, 10.0, 20.0)
MINIMO_K = 0.90
CAMINHO_SAIDA_K = RAIZ / "data" / "concorrencia_efetiva_k.csv"
CAMINHO_DIRECAO_K = RAIZ / "data" / "concorrencia_efetiva_k_direcao.csv"
COLUNAS_K = ("k", *COLUNAS)
COLUNAS_DIRECAO_K = (
    "efeito", "n_fan_out", "concorrencia", "xi", "dir_k5", "dir_k10", "dir_k20",
    "igual_k5", "igual_k20",
)


def calcular_ks(linhas_indice, ks=KS_SENSIBILIDADE, pasta_raw=PASTA_RAW, limiares=None,
                verificar_hash=True, progresso=None):
    """Como `calcular`, para vários K de uma vez (cada bruto é lido uma só vez)."""
    limiares = limiares if limiares is not None else carregar_limiares()
    inicio, total = time.monotonic(), len(linhas_indice)
    saida = {k: [] for k in ks}
    for i, linha in enumerate(linhas_indice, start=1):
        tabela = carregar_execucao(linha, pasta_raw, verificar_hash=verificar_hash)
        lim99 = limiares_do_xi(limiares, linha["xi"])["p99"]
        for k in ks:
            m = medidas_execucao(tabela, linha["concorrencia"], lim99, k=k)
            saida[k].append({"id": linha["id"], "n_fan_out": linha["n_fan_out"],
                             "concorrencia": linha["concorrencia"], "xi": linha["xi"],
                             "replicacao": linha["replicacao"], **m})
        if progresso is not None and (i % 100 == 0 or i == total):
            progresso(f"  {i}/{total} execuções ({time.monotonic() - inicio:.0f} s)")
    return saida


def direcao(medias: list[float], ics: list[float]) -> str:
    """'cresce', 'decresce' ou 'nenhuma', pela regra de `hipotese.cresce` (série já ordenada)."""
    if cresce(medias, ics)["cresce"]:
        return "cresce"
    if cresce([-m for m in medias], ics)["cresce"]:
        return "decresce"
    return "nenhuma"


def series_direcao(tabelas_k: dict) -> list[dict]:
    """Direção de f_preso com C (por N e ξ) e com ξ (por N e C), para cada K."""
    ks = sorted(tabelas_k)
    idx = {k: {(x["n_fan_out"], x["concorrencia"], x["xi"]): x for x in tabelas_k[k]} for k in ks}
    base = idx[ks[0]]
    ns = sorted({c[0] for c in base})
    cs = sorted({c[1] for c in base})
    xis = sorted({c[2] for c in base})

    def dirs(chaves):
        r = {}
        for k in ks:
            linhas = [idx[k][c] for c in chaves]
            r[k] = direcao([x["f_preso_media"] for x in linhas], [x["f_preso_ic95"] for x in linhas])
        return r

    saida = []
    for n in ns:
        for xi in xis:
            saida.append(("C", n, None, xi, dirs([(n, c, xi) for c in cs])))
    for n in ns:
        for c in cs:
            saida.append(("xi", n, c, None, dirs([(n, c, xi) for xi in xis])))
    ref = 10.0
    linhas = []
    for efeito, n, c, xi, d in saida:
        linhas.append({
            "efeito": efeito, "n_fan_out": n,
            "concorrencia": "" if c is None else c, "xi": "" if xi is None else xi,
            "dir_k5": d[5.0], "dir_k10": d[ref], "dir_k20": d[20.0],
            "igual_k5": d[5.0] == d[ref], "igual_k20": d[20.0] == d[ref],
        })
    return linhas


def veredito_k(direcoes: list[dict]) -> list[dict]:
    saida = []
    for k in (5, 20):
        testes = len(direcoes)
        passam = sum(1 for d in direcoes if d[f"igual_k{k}"])
        fracao = passam / testes
        saida.append({"k": k, "testes": testes, "passam": passam, "fracao": fracao,
                      "aprovado": fracao >= MINIMO_K - 1e-12})
    return saida


def conferir_k10(tabela_k10: list[dict], caminho=CAMINHO_SAIDA) -> int:
    """O K = 10 refeito tem de reproduzir o CSV existente (não é alterado). Devolve nº de cenários."""
    with open(caminho, newline="", encoding="utf-8") as f:
        existente = {r["id"]: r for r in csv.DictReader(f)}
    if set(existente) != {x["id"] for x in tabela_k10}:
        raise ValueError("cenários do K = 10 diferem do CSV existente")
    for x in tabela_k10:
        e = existente[x["id"]]
        for m in MEDIDAS:
            for s in ("media", "ic95"):
                a, b = float(e[f"{m}_{s}"]), float(x[f"{m}_{s}"])
                if abs(a - b) > 1e-9 * max(1.0, abs(a)):
                    raise ValueError(f"{x['id']} {m}_{s}: K = 10 refeito ({b}) != existente ({a})")
    return len(existente)


def gerar_sensibilidade_k() -> int:
    linhas = ler_indice()
    por_k = calcular_ks(linhas, progresso=print)
    cap = capacidade_da_grade()
    tabela_total, tabelas = [], {}
    for k, execs in por_k.items():
        tab = agregar(execs, cap)
        tabelas[k] = tab
        tabela_total.extend({"k": k, **x} for x in tab)
    n = conferir_k10(tabelas[10.0])
    print(f"\nK = 10 refeito reproduz data/concorrencia_efetiva.csv ({n} cenários)")
    gravar_csv(tabela_total, COLUNAS_K, CAMINHO_SAIDA_K)
    dirs = series_direcao(tabelas)
    gravar_csv(dirs, COLUNAS_DIRECAO_K, CAMINHO_DIRECAO_K)
    ver = veredito_k(dirs)
    for efeito in ("C", "xi"):
        sub = [d for d in dirs if d["efeito"] == efeito]
        cont = {k: {} for k in ("dir_k5", "dir_k10", "dir_k20")}
        for d in sub:
            for k in cont:
                cont[k][d[k]] = cont[k].get(d[k], 0) + 1
        print(f"  efeito de {efeito} ({len(sub)} séries): " +
              "; ".join(f"{k[4:]} {cont[k]}" for k in cont))
    print(f"\n{'K':>3s} {'passam':>9s} {'fração':>7s}  aprovado")
    for v in ver:
        print(f"{v['k']:3d} {v['passam']:4d}/{v['testes']:<4d} {v['fracao']:7.3f}  "
              f"{'sim' if v['aprovado'] else 'NÃO'}")
    print(f"\ngravados: {CAMINHO_SAIDA_K.name}, {CAMINHO_DIRECAO_K.name}")
    return 0 if all(v["aprovado"] for v in ver) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Concorrência efetiva (Fase 6, item 9)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--sensibilidade-k", action="store_true", dest="sens_k",
                    help="verificação B: refaz com K = 5, 10 e 20 e compara as direções")
    ap.add_argument("--saida", default=str(CAMINHO_SAIDA))
    ap.add_argument("--ids", nargs="*", default=None, help="restringe a estes cenários (teste rápido)")
    a = ap.parse_args()
    if a.sens_k:
        return gerar_sensibilidade_k()
    if not a.gerar:
        ap.error("use --gerar ou --sensibilidade-k")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
