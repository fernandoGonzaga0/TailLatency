"""Análises complementares — Etapa 8 (Fase 6, item 7). Descritivo: sem aprovação.

Usa só `data/metricas_cenarios.csv` e a teoria 1 − q^N (módulo `teoria`).

Três tabelas (todas versionadas):

1. `data/excesso_violacao.csv` — uma linha por cenário com contenção e limiar
   (80 x 3 = 240). Excesso sobre a teoria em duas escalas:
   - absoluto: dif = simulado − teoria (pontos de proporção) e IC95;
   - razão: razao = simulado / teoria e IC95 (a teoria é constante, então o IC da
     razão é o IC do simulado dividido pela teoria);
   e uma terceira, que ajuda a ler o efeito teto: folga = 1 − teoria e
   frac_folga = dif / folga (fração do espaço que ainda existia até 100%).
2. `data/excesso_por_n_c.csv` — a mesma medida média sobre os 4 valores de ξ, por
   (limiar, N, C). O IC da média de 4 cenários independentes é a raiz da soma dos
   quadrados dos IC dividida por 4.
3. `data/efeito_xi.csv` — efeito do peso da cauda sobre o excesso: por (limiar, N, C),
   diferença do excesso absoluto entre cada ξ e ξ = 0,2 (o menos pesado), com IC pela
   raiz da soma dos quadrados (cenários independentes) e a marca de significância
   (|diferença| > IC combinado).

Interação N x C x ξ: as tabelas 2 e 3 permitem ver, no texto, como o excesso muda
com C em cada N e como o efeito de ξ depende de N e de C.

    python -m src.analysis.complementar --gerar
    python -m src.analysis.teste_complementar
"""

import argparse
import math

from src.analysis.agregacao import gravar_csv
from src.analysis.carga import RAIZ
from src.analysis.metricas import NOMES_LIMIAR
from src.analysis.teoria import CAMINHO_CENARIOS, Q_VIOLACAO, ler_tabela_cenarios, violacao_teorica

PASTA = RAIZ / "data"
COLUNAS_EXCESSO = (
    "id", "limiar", "n_fan_out", "concorrencia", "xi", "teoria", "simulado",
    "dif", "dif_ic95", "razao", "razao_ic95", "folga", "frac_folga",
)
COLUNAS_N_C = (
    "limiar", "n_fan_out", "concorrencia", "teoria", "simulado", "dif", "dif_ic95",
    "razao", "razao_ic95", "folga", "frac_folga",
)
COLUNAS_XI = (
    "limiar", "n_fan_out", "concorrencia", "xi", "xi_base", "dif_xi", "dif_xi_ic95", "significativo",
)
XI_BASE_PADRAO = None  # menor ξ da tabela


def excesso_por_cenario(cenarios: list[dict]) -> list[dict]:
    saida = []
    for c in sorted(cenarios, key=lambda x: x["id"]):
        if c["sem_contencao"]:
            continue
        for nome in NOMES_LIMIAR:
            teoria = violacao_teorica(c["n_fan_out"], Q_VIOLACAO[nome])
            sim, ic = c[f"violacao_{nome}_media"], c[f"violacao_{nome}_ic95"]
            folga = 1.0 - teoria
            dif = sim - teoria
            saida.append({
                "id": c["id"], "limiar": nome, "n_fan_out": c["n_fan_out"],
                "concorrencia": c["concorrencia"], "xi": c["xi"],
                "teoria": teoria, "simulado": sim, "dif": dif, "dif_ic95": ic,
                "razao": sim / teoria, "razao_ic95": ic / teoria,
                "folga": folga, "frac_folga": dif / folga,
            })
    return saida


def _media_ic(valores_ic: list[float]) -> float:
    """IC95 da média de k estimativas independentes, dado o IC95 de cada uma."""
    k = len(valores_ic)
    return math.sqrt(sum(e * e for e in valores_ic)) / k


def excesso_por_n_c(excesso: list[dict]) -> list[dict]:
    grupos: dict[tuple, list[dict]] = {}
    for x in excesso:
        grupos.setdefault((x["limiar"], x["n_fan_out"], x["concorrencia"]), []).append(x)
    saida = []
    for (limiar, n, c), g in sorted(grupos.items(), key=lambda kv: (NOMES_LIMIAR.index(kv[0][0]), kv[0][1], kv[0][2])):
        k = len(g)
        teoria = g[0]["teoria"]
        sim = sum(x["simulado"] for x in g) / k
        ic = _media_ic([x["dif_ic95"] for x in g])
        folga = 1.0 - teoria
        saida.append({
            "limiar": limiar, "n_fan_out": n, "concorrencia": c, "teoria": teoria,
            "simulado": sim, "dif": sim - teoria, "dif_ic95": ic,
            "razao": sim / teoria, "razao_ic95": ic / teoria,
            "folga": folga, "frac_folga": (sim - teoria) / folga,
        })
    return saida


def efeito_xi(excesso: list[dict], xi_base: float | None = None) -> list[dict]:
    xis = sorted({x["xi"] for x in excesso})
    base = xis[0] if xi_base is None else xi_base
    if base not in xis:
        raise ValueError(f"xi_base {base} não está na tabela")
    idx = {(x["limiar"], x["n_fan_out"], x["concorrencia"], x["xi"]): x for x in excesso}
    saida = []
    for (limiar, n, c, xi), x in sorted(
        idx.items(), key=lambda kv: (NOMES_LIMIAR.index(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])
    ):
        if xi == base:
            continue
        b = idx[(limiar, n, c, base)]
        dif = x["dif"] - b["dif"]
        ic = math.hypot(x["dif_ic95"], b["dif_ic95"])
        saida.append({
            "limiar": limiar, "n_fan_out": n, "concorrencia": c, "xi": xi, "xi_base": base,
            "dif_xi": dif, "dif_xi_ic95": ic, "significativo": abs(dif) > ic,
        })
    return saida


def analisar(cenarios: list[dict]) -> dict:
    exc = excesso_por_cenario(cenarios)
    return {"excesso": exc, "n_c": excesso_por_n_c(exc), "xi": efeito_xi(exc)}


def gerar(caminho_cenarios, pasta=PASTA) -> dict:
    res = analisar(ler_tabela_cenarios(caminho_cenarios))
    gravar_csv(res["excesso"], COLUNAS_EXCESSO, pasta / "excesso_violacao.csv")
    gravar_csv(res["n_c"], COLUNAS_N_C, pasta / "excesso_por_n_c.csv")
    gravar_csv(res["xi"], COLUNAS_XI, pasta / "efeito_xi.csv")
    return res


def _imprimir(res: dict) -> None:
    print("EXCESSO MÉDIO SOBRE 1−q^N (média dos 4 ξ): razão simulado/teoria [dif em pontos de proporção]")
    for nome in NOMES_LIMIAR:
        linhas = [x for x in res["n_c"] if x["limiar"] == nome]
        cs = sorted({x["concorrencia"] for x in linhas})
        print(f"\n {nome}   " + "".join(f"{'C=' + str(c):>16s}" for c in cs))
        for n in sorted({x["n_fan_out"] for x in linhas}):
            cel = {x["concorrencia"]: x for x in linhas if x["n_fan_out"] == n}
            print(f"  N={n:2d}  " + "".join(
                f"{cel[c]['razao']:8.3f}[{100 * cel[c]['dif']:+6.2f}]" for c in cs))
    print("\nEFEITO DE ξ (vs menor ξ) no excesso absoluto: casos significativos / total")
    for nome in NOMES_LIMIAR:
        s = [x for x in res["xi"] if x["limiar"] == nome]
        for xi in sorted({x["xi"] for x in s}):
            t = [x for x in s if x["xi"] == xi]
            pos = sum(1 for x in t if x["significativo"] and x["dif_xi"] > 0)
            neg = sum(1 for x in t if x["significativo"] and x["dif_xi"] < 0)
            print(f"  {nome} xi={xi}: {pos + neg:2d}/{len(t)} significativos (maior excesso: {pos}, menor: {neg})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Análises complementares (Fase 6, item 7)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--cenarios", default=str(CAMINHO_CENARIOS))
    ap.add_argument("--pasta", default=str(PASTA))
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    from pathlib import Path

    res = gerar(a.cenarios, Path(a.pasta))
    _imprimir(res)
    print(f"\ngravado em {a.pasta}: excesso_violacao.csv, excesso_por_n_c.csv, efeito_xi.csv")


if __name__ == "__main__":
    main()
