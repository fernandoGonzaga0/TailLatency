"""Verificação da hipótese — Etapa 8 (Fase 6, item 6).

Hipótese: sem contenção a violação é 1 − q^N e cresce com o fan-out N; com
concorrência ela fica acima disso e cresce com C.

Usa só `data/metricas_cenarios.csv` (médias e IC t de 95% entre as 10 replicações).
Critérios fixados em 14/11/2026, antes da execução deste módulo. Nota de
transparência: o critério "acima" foi fixado depois de ver a prévia do item 3
(p90 72/80, p95 76/80, p99 68/80 cenários com o IC acima da teoria); ele não foi
ajustado depois disso.

Regra de crescimento de uma série (valores m_i com semilargura e_i, ordenados
pela variável de interesse — C ou N):

  (i)  nenhum passo consecutivo cai de forma significativa: m_i − m_(i+1) <= e_i + e_(i+1);
  (ii) o último ponto fica significativamente acima do primeiro:
       m_último − e_último > m_primeiro + e_primeiro.

(Critério conservador: comparar IC por sobreposição é mais exigente que testar a
diferença; uma série que passa por ele é robusta.)

Parte 1 (sem contenção): (a) a violação bate com 1 − q^N — critério do item 3 (já
aprovado, 4/120 fora do IC); (b) cresce com N: série em N = 1, 5, 10, 20, 50 pela
regra acima, para cada ξ, limiar e C <= capacidade (24 séries); aprovada se pelo
menos 90% passam. (O item (b) é acréscimo meu, sujeito a veto.)

Parte 2 (com contenção): (a) acima da teoria: o IC inteiro acima de 1 − q^N em
pelo menos 90% dos 80 cenários, por limiar; (b) cresce com C: série em C = capacidade
(sem contenção, ponto de partida) e nos valores com contenção, pela regra acima, para
cada N, ξ e limiar (60 séries); aprovada se pelo menos 90% (54) passam. Casos que
falham são listados, não escondidos; o teto (violação já próxima de 100%) e o N
pequeno são causas esperadas e devem ser verificadas caso a caso.

Saídas (versionadas): data/hipotese_crescimento.csv (uma linha por série das
partes 1b e 2b) e data/hipotese_veredito.csv (uma linha por critério).

    python -m src.analysis.hipotese --gerar
    python -m src.analysis.teste_hipotese
"""

import argparse
from pathlib import Path

from src.analysis.agregacao import gravar_csv
from src.analysis.carga import RAIZ
from src.analysis.metricas import NOMES_LIMIAR
from src.analysis.teoria import (
    CAMINHO_CENARIOS,
    Q_VIOLACAO,
    ler_tabela_cenarios,
    posicao,
    violacao_teorica,
)

CAMINHO_CRESCIMENTO = RAIZ / "data" / "hipotese_crescimento.csv"
CAMINHO_VEREDITO = RAIZ / "data" / "hipotese_veredito.csv"
FRACAO_MINIMA = 0.90

COLUNAS_CRESCIMENTO = (
    "parte", "variavel", "limiar", "xi", "n_fan_out", "concorrencia",
    "valores_x", "medias", "quedas_significativas", "ganho_significativo", "cresce",
)
COLUNAS_VEREDITO = ("criterio", "limiar", "testes", "passam", "fracao", "minimo", "aprovado")


def cresce(medias: list[float], ics: list[float]) -> dict:
    """Aplica a regra de crescimento a uma série já ordenada."""
    if len(medias) < 2 or len(medias) != len(ics):
        raise ValueError("série precisa de pelo menos 2 pontos, com um IC por ponto")
    quedas = sum(
        1 for i in range(len(medias) - 1)
        if medias[i] - medias[i + 1] > ics[i] + ics[i + 1]
    )
    ganho = (medias[-1] - ics[-1]) > (medias[0] + ics[0])
    return {"quedas": quedas, "ganho": ganho, "cresce": quedas == 0 and ganho}


def _capacidade(cenarios: list[dict]) -> int:
    return max(c["concorrencia"] for c in cenarios if c["sem_contencao"])


def _indexar(cenarios):
    return {(c["n_fan_out"], c["concorrencia"], c["xi"]): c for c in cenarios}


def _limiares():
    return [(nome, f"violacao_{nome}", Q_VIOLACAO[nome]) for nome in NOMES_LIMIAR]


def series_em_n(cenarios: list[dict]) -> list[dict]:
    """Parte 1b: crescimento com N, em cada ξ, limiar e C sem contenção."""
    idx = _indexar(cenarios)
    cap = _capacidade(cenarios)
    ns = sorted({c["n_fan_out"] for c in cenarios})
    cs = sorted({c["concorrencia"] for c in cenarios if c["concorrencia"] <= cap})
    xis = sorted({c["xi"] for c in cenarios})
    saida = []
    for nome, col, _ in _limiares():
        for xi in xis:
            for conc in cs:
                linhas = [idx[(n, conc, xi)] for n in ns]
                r = cresce([x[f"{col}_media"] for x in linhas], [x[f"{col}_ic95"] for x in linhas])
                saida.append({
                    "parte": "1b", "variavel": "N", "limiar": nome, "xi": xi,
                    "n_fan_out": "", "concorrencia": conc,
                    "valores_x": " ".join(str(n) for n in ns),
                    "medias": " ".join(f"{x[f'{col}_media']:.4f}" for x in linhas),
                    "quedas_significativas": r["quedas"], "ganho_significativo": r["ganho"],
                    "cresce": r["cresce"],
                })
    return saida


def series_em_c(cenarios: list[dict]) -> list[dict]:
    """Parte 2b: crescimento com C, de C = capacidade em diante, em cada N, ξ e limiar."""
    idx = _indexar(cenarios)
    cap = _capacidade(cenarios)
    cs = sorted({c["concorrencia"] for c in cenarios if c["concorrencia"] >= cap})
    ns = sorted({c["n_fan_out"] for c in cenarios})
    xis = sorted({c["xi"] for c in cenarios})
    saida = []
    for nome, col, _ in _limiares():
        for n in ns:
            for xi in xis:
                linhas = [idx[(n, conc, xi)] for conc in cs]
                r = cresce([x[f"{col}_media"] for x in linhas], [x[f"{col}_ic95"] for x in linhas])
                saida.append({
                    "parte": "2b", "variavel": "C", "limiar": nome, "xi": xi,
                    "n_fan_out": n, "concorrencia": "",
                    "valores_x": " ".join(str(c) for c in cs),
                    "medias": " ".join(f"{x[f'{col}_media']:.4f}" for x in linhas),
                    "quedas_significativas": r["quedas"], "ganho_significativo": r["ganho"],
                    "cresce": r["cresce"],
                })
    return saida


def casos_acima(cenarios: list[dict]) -> list[dict]:
    """Parte 2a: para cada cenário com contenção e limiar, o IC inteiro está acima de 1 − q^N?"""
    saida = []
    for c in sorted(cenarios, key=lambda x: x["id"]):
        if c["sem_contencao"]:
            continue
        for nome, col, q in _limiares():
            teoria = violacao_teorica(c["n_fan_out"], q)
            pos = posicao(teoria, c[f"{col}_media"], c[f"{col}_ic95"])
            saida.append({"id": c["id"], "limiar": nome, "n_fan_out": c["n_fan_out"],
                          "concorrencia": c["concorrencia"], "xi": c["xi"],
                          "teoria": teoria, "simulado": c[f"{col}_media"], "posicao": pos})
    return saida


def _linha_veredito(criterio, limiar, testes, passam, minimo=FRACAO_MINIMA):
    fracao = passam / testes if testes else 0.0
    return {"criterio": criterio, "limiar": limiar, "testes": testes, "passam": passam,
            "fracao": round(fracao, 4), "minimo": minimo, "aprovado": fracao >= minimo}


def avaliar(cenarios: list[dict]) -> dict:
    """Roda os critérios das partes 1b, 2a e 2b."""
    em_n, em_c, acima = series_em_n(cenarios), series_em_c(cenarios), casos_acima(cenarios)
    veredito = []
    for nome in NOMES_LIMIAR:
        s = [x for x in em_n if x["limiar"] == nome]
        veredito.append(_linha_veredito("1b_cresce_com_N", nome, len(s), sum(x["cresce"] for x in s)))
    for nome in NOMES_LIMIAR:
        s = [x for x in acima if x["limiar"] == nome]
        veredito.append(_linha_veredito("2a_acima_da_teoria", nome, len(s),
                                        sum(x["posicao"] == "acima" for x in s)))
    for nome in NOMES_LIMIAR:
        s = [x for x in em_c if x["limiar"] == nome]
        veredito.append(_linha_veredito("2b_cresce_com_C", nome, len(s), sum(x["cresce"] for x in s)))
    return {"series": em_n + em_c, "acima": acima, "veredito": veredito}


def falhas(res: dict) -> dict:
    return {
        "2a": [x for x in res["acima"] if x["posicao"] != "acima"],
        "series": [x for x in res["series"] if not x["cresce"]],
    }


def _imprimir(res: dict) -> None:
    print("VEREDITO (mínimo 90% por critério e por limiar)")
    for v in res["veredito"]:
        marca = "APROVADO" if v["aprovado"] else "NÃO aprovado"
        print(f"  {v['criterio']:20s} {v['limiar']}: {v['passam']:3d}/{v['testes']:3d}"
              f" ({100 * v['fracao']:5.1f}%)  {marca}")
    f = falhas(res)
    if f["2a"]:
        print("\nCASOS 2a em que o IC NÃO ficou inteiro acima da teoria (limiar, N, C, ξ):")
        for x in f["2a"]:
            print(f"  {x['limiar']} N={x['n_fan_out']:2d} C={x['concorrencia']:2d} xi={x['xi']}: "
                  f"simulado {x['simulado']:.4f} vs teoria {x['teoria']:.4f} ({x['posicao']})")
    if f["series"]:
        print("\nSÉRIES que NÃO crescem (parte, limiar, N/C, ξ, quedas, ganho):")
        for x in f["series"]:
            print(f"  {x['parte']} {x['limiar']} N={x['n_fan_out']!s:2} C={x['concorrencia']!s:2} "
                  f"xi={x['xi']}: quedas={x['quedas_significativas']} ganho={x['ganho_significativo']}"
                  f" | x={x['valores_x']} | m={x['medias']}")


def gerar(caminho_cenarios, saida_crescimento, saida_veredito) -> dict:
    res = avaliar(ler_tabela_cenarios(caminho_cenarios))
    gravar_csv(res["series"], COLUNAS_CRESCIMENTO, saida_crescimento)
    gravar_csv(res["veredito"], COLUNAS_VEREDITO, saida_veredito)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Verificação da hipótese (Fase 6, item 6)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--cenarios", default=str(CAMINHO_CENARIOS))
    ap.add_argument("--crescimento", default=str(CAMINHO_CRESCIMENTO))
    ap.add_argument("--veredito", default=str(CAMINHO_VEREDITO))
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    res = gerar(a.cenarios, a.crescimento, a.veredito)
    _imprimir(res)
    print(f"\ngravado: {a.crescimento}\ngravado: {a.veredito}")
    raise SystemExit(0 if all(v["aprovado"] for v in res["veredito"]) else 1)


if __name__ == "__main__":
    main()
