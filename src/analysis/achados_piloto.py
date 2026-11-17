"""Achados do piloto, verificados na grade completa — Fase 6, item 9. Descritivo.

Usa só `data/metricas_cenarios.csv`.

1. p99/p50 do conjunto cai com a concorrência? Por (N, ξ), de C = capacidade até o
   maior C: `decresce` = nenhum passo consecutivo sobe de forma significativa e o
   último ponto fica significativamente abaixo do primeiro (a regra de
   `hipotese.cresce` aplicada aos valores negados). Para mostrar o mecanismo, a
   tabela traz a razão C máximo / C = capacidade do p50 e do p99 do conjunto: o p99/p50
   cai quando o p50 sobe proporcionalmente mais que o p99 (`mediana_sobe_mais`).
   Saída: data/amplificacao_vs_c.csv (20 linhas).
2. Efeito teto: um par (N, limiar) está "no teto" se a violação teórica 1 − q^N é
   de pelo menos 0,95 (folga <= 0,05): ali o excesso sobre a teoria não pode
   passar de 5 pontos de proporção, e a hipótese 2a perde poder de detecção.
   Saída: data/teto.csv (15 linhas).

    python -m src.analysis.achados_piloto --gerar
    python -m src.analysis.teste_achados_piloto
"""

import argparse
from pathlib import Path

from src.analysis.agregacao import gravar_csv
from src.analysis.carga import RAIZ
from src.analysis.hipotese import cresce
from src.analysis.metricas import NOMES_LIMIAR
from src.analysis.teoria import CAMINHO_CENARIOS, Q_VIOLACAO, ler_tabela_cenarios, violacao_teorica

PASTA = RAIZ / "data"
LIMITE_TETO = 0.95
COLUNAS_AMP = (
    "n_fan_out", "xi", "c_inicial", "c_final", "amp_inicial", "amp_final", "razao_amp",
    "p50_razao", "p99_razao", "mediana_sobe_mais", "decresce",
)
COLUNAS_TETO = ("n_fan_out", "limiar", "teoria", "folga", "teto")


def amplificacao_vs_c(cenarios: list[dict]) -> list[dict]:
    cap = max(c["concorrencia"] for c in cenarios if c["sem_contencao"])
    idx = {(c["n_fan_out"], c["concorrencia"], c["xi"]): c for c in cenarios}
    cs = sorted({c["concorrencia"] for c in cenarios if c["concorrencia"] >= cap})
    saida = []
    for n in sorted({c["n_fan_out"] for c in cenarios}):
        for xi in sorted({c["xi"] for c in cenarios}):
            linhas = [idx[(n, c, xi)] for c in cs]
            med = [x["amplificacao_p99_p50_media"] for x in linhas]
            ics = [x["amplificacao_p99_p50_ic95"] for x in linhas]
            r = cresce([-m for m in med], ics)
            p50 = linhas[-1]["p50_ms_media"] / linhas[0]["p50_ms_media"]
            p99 = linhas[-1]["p99_ms_media"] / linhas[0]["p99_ms_media"]
            saida.append({
                "n_fan_out": n, "xi": xi, "c_inicial": cs[0], "c_final": cs[-1],
                "amp_inicial": med[0], "amp_final": med[-1], "razao_amp": med[-1] / med[0],
                "p50_razao": p50, "p99_razao": p99, "mediana_sobe_mais": p50 > p99,
                "decresce": r["cresce"],
            })
    return saida


def tabela_teto(cenarios: list[dict]) -> list[dict]:
    saida = []
    for n in sorted({c["n_fan_out"] for c in cenarios}):
        for nome in NOMES_LIMIAR:
            t = violacao_teorica(n, Q_VIOLACAO[nome])
            saida.append({"n_fan_out": n, "limiar": nome, "teoria": t, "folga": 1.0 - t,
                          "teto": t >= LIMITE_TETO})
    return saida


def gerar(caminho_cenarios, pasta=PASTA) -> dict:
    cen = ler_tabela_cenarios(caminho_cenarios)
    res = {"amp": amplificacao_vs_c(cen), "teto": tabela_teto(cen)}
    gravar_csv(res["amp"], COLUNAS_AMP, pasta / "amplificacao_vs_c.csv")
    gravar_csv(res["teto"], COLUNAS_TETO, pasta / "teto.csv")
    return res


def _imprimir(res: dict) -> None:
    a = res["amp"]
    print(f"p99/p50 DO CONJUNTO vs C ({a[0]['c_inicial']} -> {a[0]['c_final']}):")
    print(f"  decresce de forma significativa em {sum(x['decresce'] for x in a)}/{len(a)} pares (N, ξ)")
    print(f"  p50 sobe proporcionalmente mais que o p99 em {sum(x['mediana_sobe_mais'] for x in a)}/{len(a)}")
    for x in a:
        print(f"  N={x['n_fan_out']:2d} xi={x['xi']}: p99/p50 {x['amp_inicial']:.2f} -> {x['amp_final']:.2f}"
              f" (x{x['razao_amp']:.2f}); p50 x{x['p50_razao']:.2f}, p99 x{x['p99_razao']:.2f}"
              f"{'  [não decresce]' if not x['decresce'] else ''}")
    print("\nTETO (violação teórica >= 0,95):")
    for x in res["teto"]:
        if x["teto"]:
            print(f"  N={x['n_fan_out']} {x['limiar']}: teoria {x['teoria']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Achados do piloto (Fase 6, item 9)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--cenarios", default=str(CAMINHO_CENARIOS))
    ap.add_argument("--pasta", default=str(PASTA))
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    _imprimir(gerar(a.cenarios, Path(a.pasta)))
    print(f"\ngravado em {a.pasta}: amplificacao_vs_c.csv, teto.csv")


if __name__ == "__main__":
    main()
