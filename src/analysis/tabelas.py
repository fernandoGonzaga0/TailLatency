"""Tabelas do texto em LaTeX — Etapa 8 (Fase 6, item 10).

Lê só as tabelas versionadas em `data/` e grava `.tex` (ambiente `tabular` com
`booktabs`) em `results/tabelas/`, com vírgula decimal. Cada arquivo é só o
`tabular`; legenda e rótulo ficam a cargo do texto, no `\\begin{table}` de quem
inclui com `\\input{...}`. O preâmbulo precisa de `\\usepackage{booktabs}`.

Tabelas:
 - tab_validacao: teoria x simulado (sem contenção: dentro do IC; com contenção:
   acima/dentro/abaixo), por métrica;
 - tab_veredito: critérios da hipótese (itens 3 e 6);
 - tab_excesso_{p90,p95,p99}: razão simulado/teoria e diferença (pontos de proporção)
   por N e C, média sobre os ξ;
 - tab_amplificacao: p99/p50 do conjunto de C = capacidade a C máximo, por N e ξ;
 - tab_concorrencia_efetiva: concorrência efetiva ± IC95 por N e ξ, em C = 4 e C = 12.

    python -m src.analysis.tabelas --gerar
    python -m src.analysis.teste_figuras
"""

import argparse
import csv
from pathlib import Path

from src.analysis.carga import RAIZ
from src.analysis.metricas import NOMES_LIMIAR
from src.analysis.teoria import METRICAS as METRICAS_TEORIA

PASTA_DATA = RAIZ / "data"
PASTA_TAB = RAIZ / "results" / "tabelas"

ROTULO_METRICA = {
    "violacao_p90": "Violação, limiar p90", "violacao_p95": "Violação, limiar p95",
    "violacao_p99": "Violação, limiar p99", "p50_ms": "p50 do conjunto",
    "p95_ms": "p95 do conjunto", "p99_ms": "p99 do conjunto",
    "amplificacao_p99_p50": "p99/p50 do conjunto",
}
ROTULO_CRITERIO = {
    "1b_cresce_com_N": "Cresce com $N$ (sem contenção)",
    "2a_acima_da_teoria": "Acima de $1-q^N$ (com contenção)",
    "2b_cresce_com_C": "Cresce com $C$ (com contenção)",
}


def num(x, casas: int = 2, sinal: bool = False) -> str:
    """Número com vírgula decimal."""
    t = f"{float(x):+.{casas}f}" if sinal else f"{float(x):.{casas}f}"
    return t.replace(".", ",")


def esc(texto: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        texto = texto.replace(a, b)
    return texto


def tabular(colunas: str, cabecalho: list[str], linhas: list[list[str]], *, antes: list[str] | None = None,
            separadores: set[int] | None = None) -> str:
    """Monta o `tabular` com booktabs. `antes` = linhas cruas entre \\toprule e o cabeçalho;
    `separadores` = índices de linha depois dos quais há \\midrule."""
    out = [f"\\begin{{tabular}}{{{colunas}}}", "\\toprule", *(antes or []), " & ".join(cabecalho) + " \\\\"]
    out.append("\\midrule")
    for i, ln in enumerate(linhas):
        out.append(" & ".join(ln) + " \\\\")
        if separadores and i in separadores and i != len(linhas) - 1:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def _ler(caminho: Path) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tab_validacao(comparacao: list[dict]) -> str:
    sem = [x for x in comparacao if x["sem_contencao"] == "true"]
    com = [x for x in comparacao if x["sem_contencao"] != "true"]
    linhas = []
    for m in METRICAS_TEORIA:
        dentro = sum(x[f"{m}_posicao"] == "dentro" for x in sem)
        c = {k: sum(x[f"{m}_posicao"] == k for x in com) for k in ("acima", "dentro", "abaixo")}
        linhas.append([ROTULO_METRICA[m], f"{dentro}/{len(sem)}", str(c["acima"]), str(c["dentro"]), str(c["abaixo"])])
    return tabular("lrrrr", ["Métrica", "Dentro do IC (de 40)", "Acima", "Dentro", "Abaixo"], linhas,
                   antes=[r" & Sem contenção & \multicolumn{3}{c}{Com contenção (80 cenários)} \\",
                          r"\cmidrule(lr){3-5}"])


def tab_veredito(veredito: list[dict]) -> str:
    linhas = []
    for v in veredito:
        pct = num(100 * float(v["fracao"]), 1)
        ok = v["aprovado"] == "True" or v["aprovado"] == "true"
        linhas.append([ROTULO_CRITERIO.get(v["criterio"], esc(v["criterio"])), esc(v["limiar"]),
                       f"{v['passam']}/{v['testes']}", f"{pct}\\%", "sim" if ok else "não"])
    return tabular("llrrl", ["Critério", "Limiar", "Casos", "Fração", "Aprovado"], linhas,
                   separadores={2, 5})


def tab_excesso(n_c: list[dict], limiar: str) -> str:
    linhas_l = [x for x in n_c if x["limiar"] == limiar]
    cs = sorted({int(x["concorrencia"]) for x in linhas_l})
    ns = sorted({int(x["n_fan_out"]) for x in linhas_l})
    idx = {(int(x["n_fan_out"]), int(x["concorrencia"])): x for x in linhas_l}
    linhas = []
    for n in ns:
        linha = [str(n)]
        for c in cs:
            x = idx[(n, c)]
            linha.append(f"{num(x['razao'], 2)} ({num(100 * float(x['dif']), 1, sinal=True)})")
        linhas.append(linha)
    return tabular("r" + "r" * len(cs), ["$N$"] + [f"$C$ = {c}" for c in cs], linhas)


def tab_amplificacao(amp: list[dict]) -> str:
    linhas, seps = [], set()
    ns = sorted({int(x["n_fan_out"]) for x in amp})
    for n in ns:
        for x in (y for y in amp if int(y["n_fan_out"]) == n):
            linhas.append([str(n), num(x["xi"], 1), num(x["amp_inicial"], 2), num(x["amp_final"], 2),
                           num(x["razao_amp"], 2), num(x["p50_razao"], 2), num(x["p99_razao"], 2),
                           "sim" if x["decresce"] in ("True", "true") else "não"])
        seps.add(len(linhas) - 1)
    c0, c1 = amp[0]["c_inicial"], amp[0]["c_final"]
    return tabular("rrrrrrrl", ["$N$", "$\\xi$", f"$C$={c0}", f"$C$={c1}", "Razão", "$p50$ (razão)",
                                "$p99$ (razão)", "Cai"], linhas, separadores=seps)


def tab_concorrencia_efetiva(linhas_ce: list[dict]) -> str:
    idx = {(int(x["n_fan_out"]), int(x["concorrencia"]), float(x["xi"])): x for x in linhas_ce}
    ns = sorted({k[0] for k in idx})
    xis = sorted({k[2] for k in idx})
    cs = (4, 12)
    k = len(xis)
    antes = [" & " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{$C$ = {c}}}" for c in cs) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}" for i in range(len(cs)))]
    cab = ["$N$"] + [f"$\\xi$={num(xi, 1)}" for _ in cs for xi in xis]
    linhas = []
    for n in ns:
        linha = [str(n)]
        for c in cs:
            for xi in xis:
                x = idx[(n, c, xi)]
                linha.append(f"{num(x['c_efetivo_media'], 1)}$\\pm${num(x['c_efetivo_ic95'], 1)}")
        linhas.append(linha)
    return tabular("r" + "r" * (len(cs) * k), cab, linhas, antes=antes)


def gerar(pasta_data: Path = PASTA_DATA, pasta_saida: Path = PASTA_TAB) -> list[Path]:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    arquivos = {
        "tab_validacao": tab_validacao(_ler(pasta_data / "comparacao_teoria.csv")),
        "tab_veredito": tab_veredito(_ler(pasta_data / "hipotese_veredito.csv")),
        "tab_amplificacao": tab_amplificacao(_ler(pasta_data / "amplificacao_vs_c.csv")),
        "tab_concorrencia_efetiva": tab_concorrencia_efetiva(_ler(pasta_data / "concorrencia_efetiva.csv")),
    }
    n_c = _ler(pasta_data / "excesso_por_n_c.csv")
    for nome in NOMES_LIMIAR:
        arquivos[f"tab_excesso_{nome}"] = tab_excesso(n_c, nome)
    saidas = []
    for nome, texto in arquivos.items():
        p = pasta_saida / f"{nome}.tex"
        p.write_text(texto, encoding="utf-8", newline="\n")
        saidas.append(p)
    return saidas


def main() -> None:
    ap = argparse.ArgumentParser(description="Tabelas LaTeX do texto (Fase 6, item 10)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--data", default=str(PASTA_DATA))
    ap.add_argument("--saida", default=str(PASTA_TAB))
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    for p in gerar(Path(a.data), Path(a.saida)):
        print(f"gravado: {p}")


if __name__ == "__main__":
    main()
