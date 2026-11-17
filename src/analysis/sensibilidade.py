"""Sensibilidade ao limiar de SLA — Etapa 8 (Fase 6, itens 7/8 do plano).

Critério de robustez fixado em 06/10/2026 (mesma direção do efeito e mesma ordem
dos cenários nos três limiares), operacionalizado em 14/11/2026, antes de rodar:

Direção. Para cada efeito e cada combinação dos demais fatores, classifica-se a
mudança da violação em "mais", "menos" ou "nulo" por limiar: diferença entre os dois
níveis maior que o IC combinado (raiz da soma dos quadrados dos IC) é "mais"/"menos";
caso contrário "nulo". Há **conflito** quando, na mesma combinação, um limiar dá
"mais" e outro dá "menos". Efeitos: N (N=50 contra N=1, para cada ξ e cada C sem
contenção), C (C máximo contra C=capacidade, para cada N e ξ) e ξ (ξ máximo contra
ξ mínimo, para cada N e C). Aprovado se não há conflito em nenhuma combinação.
"Nulo" em um limiar e "mais" em outro não é conflito (é perda de poder, como no teto).

Ordem. Correlação de postos de Spearman da violação média entre cada par de limiares
(p90-p95, p95-p99, p90-p99): (a) nos 120 cenários; (b) dentro de cada N (24 cenários
de C x ξ), porque a ordem global é dominada por N. Aprovado se todos os rho >= 0,90.
Empates (efeito teto) entram como postos médios e a quantidade de empates é
registrada, não escondida.

Saídas versionadas: data/sensibilidade_direcao.csv e data/sensibilidade_ordem.csv.

    python -m src.analysis.sensibilidade --gerar
    python -m src.analysis.teste_sensibilidade
"""

import argparse
import math
from itertools import combinations
from pathlib import Path

from scipy.stats import spearmanr

from src.analysis.agregacao import gravar_csv
from src.analysis.carga import RAIZ
from src.analysis.metricas import NOMES_LIMIAR
from src.analysis.teoria import CAMINHO_CENARIOS, ler_tabela_cenarios

PASTA = RAIZ / "data"
RHO_MINIMO = 0.90
COLUNAS_DIRECAO = ("efeito", "n_fan_out", "concorrencia", "xi", "dir_p90", "dir_p95", "dir_p99", "conflito")
COLUNAS_ORDEM = ("escopo", "par", "n_cenarios", "rho", "empates", "minimo", "aprovado")


def direcao(media_a: float, ic_a: float, media_b: float, ic_b: float) -> str:
    """Classifica a mudança de A para B: 'mais', 'menos' ou 'nulo'."""
    dif, ic = media_b - media_a, math.hypot(ic_a, ic_b)
    return "mais" if dif > ic else ("menos" if dif < -ic else "nulo")


def _idx(cenarios):
    return {(c["n_fan_out"], c["concorrencia"], c["xi"]): c for c in cenarios}


def _linha(efeito, n, c, xi, a, b):
    d = {nome: direcao(a[f"violacao_{nome}_media"], a[f"violacao_{nome}_ic95"],
                       b[f"violacao_{nome}_media"], b[f"violacao_{nome}_ic95"]) for nome in NOMES_LIMIAR}
    conflito = "mais" in d.values() and "menos" in d.values()
    return {"efeito": efeito, "n_fan_out": n, "concorrencia": c, "xi": xi,
            **{f"dir_{k}": v for k, v in d.items()}, "conflito": conflito}


def tabela_direcao(cenarios: list[dict]) -> list[dict]:
    idx = _idx(cenarios)
    cap = max(c["concorrencia"] for c in cenarios if c["sem_contencao"])
    ns = sorted({c["n_fan_out"] for c in cenarios})
    cs = sorted({c["concorrencia"] for c in cenarios})
    xis = sorted({c["xi"] for c in cenarios})
    saida = []
    for xi in xis:
        for c in (x for x in cs if x <= cap):
            saida.append(_linha("N", "", c, xi, idx[(ns[0], c, xi)], idx[(ns[-1], c, xi)]))
    for n in ns:
        for xi in xis:
            saida.append(_linha("C", n, "", xi, idx[(n, cap, xi)], idx[(n, cs[-1], xi)]))
    for n in ns:
        for c in cs:
            saida.append(_linha("xi", n, c, "", idx[(n, c, xis[0])], idx[(n, c, xis[-1])]))
    return saida


def _rho(cen: list[dict], a: str, b: str) -> tuple[float, int]:
    va = [c[f"violacao_{a}_media"] for c in cen]
    vb = [c[f"violacao_{b}_media"] for c in cen]
    rho = float(spearmanr(va, vb).statistic)
    empates = (len(va) - len(set(va))) + (len(vb) - len(set(vb)))
    return rho, empates


def tabela_ordem(cenarios: list[dict], minimo: float = RHO_MINIMO) -> list[dict]:
    pares = list(combinations(NOMES_LIMIAR, 2))
    escopos = [("global", cenarios)] + [
        (f"N={n}", [c for c in cenarios if c["n_fan_out"] == n])
        for n in sorted({c["n_fan_out"] for c in cenarios})
    ]
    saida = []
    for nome, cen in escopos:
        for a, b in pares:
            rho, emp = _rho(cen, a, b)
            saida.append({"escopo": nome, "par": f"{a}-{b}", "n_cenarios": len(cen), "rho": rho,
                          "empates": emp, "minimo": minimo, "aprovado": rho >= minimo})
    return saida


def avaliar(cenarios: list[dict]) -> dict:
    d, o = tabela_direcao(cenarios), tabela_ordem(cenarios)
    return {"direcao": d, "ordem": o,
            "direcao_ok": not any(x["conflito"] for x in d),
            "ordem_ok": all(x["aprovado"] for x in o)}


def gerar(caminho_cenarios, pasta=PASTA) -> dict:
    res = avaliar(ler_tabela_cenarios(caminho_cenarios))
    gravar_csv(res["direcao"], COLUNAS_DIRECAO, Path(pasta) / "sensibilidade_direcao.csv")
    gravar_csv(res["ordem"], COLUNAS_ORDEM, Path(pasta) / "sensibilidade_ordem.csv")
    return res


def _imprimir(res: dict) -> None:
    d = res["direcao"]
    print("DIREÇÃO (conflito = um limiar 'mais' e outro 'menos'):")
    for efeito in ("N", "C", "xi"):
        s = [x for x in d if x["efeito"] == efeito]
        cont = {k: sum(1 for x in s for n in NOMES_LIMIAR if x[f"dir_{n}"] == k) for k in ("mais", "nulo", "menos")}
        print(f"  efeito de {efeito:2s}: {len(s):2d} combinações, conflitos {sum(x['conflito'] for x in s)}"
              f"  | classificações (3 limiares): mais {cont['mais']}, nulo {cont['nulo']}, menos {cont['menos']}")
    for x in d:
        if x["conflito"]:
            print(f"    CONFLITO {x['efeito']} N={x['n_fan_out']} C={x['concorrencia']} xi={x['xi']}: "
                  f"p90 {x['dir_p90']}, p95 {x['dir_p95']}, p99 {x['dir_p99']}")
    print(f"  => {'APROVADO' if res['direcao_ok'] else 'NÃO aprovado'}")
    print(f"\nORDEM (Spearman, mínimo {RHO_MINIMO}):")
    for x in res["ordem"]:
        print(f"  {x['escopo']:7s} {x['par']:8s} n={x['n_cenarios']:3d}  rho={x['rho']:.4f}  "
              f"empates={x['empates']:3d}  {'ok' if x['aprovado'] else 'ABAIXO DO MÍNIMO'}")
    print(f"  => {'APROVADO' if res['ordem_ok'] else 'NÃO aprovado'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensibilidade ao limiar (Fase 6)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--cenarios", default=str(CAMINHO_CENARIOS))
    ap.add_argument("--pasta", default=str(PASTA))
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    res = gerar(a.cenarios, a.pasta)
    _imprimir(res)
    print(f"\ngravado em {a.pasta}: sensibilidade_direcao.csv, sensibilidade_ordem.csv")
    raise SystemExit(0 if (res["direcao_ok"] and res["ordem_ok"]) else 1)


if __name__ == "__main__":
    main()
