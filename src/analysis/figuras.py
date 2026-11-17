"""Figuras do texto — Etapa 8 (Fase 6, item 4).

Lê só as tabelas versionadas em `data/` (não precisa do dado bruto) e grava PDFs
vetoriais e PNGs em `results/figuras/` (a pasta `results/` não vai para o Git; as
figuras se regeneram com um comando). Fontes embutidas (TrueType) para o LaTeX.

Convenções (método de visualização de dados do projeto):
- ξ é ordinal: azul de uma matiz só, do claro ao escuro (validado: lightness
  monótona, passos visíveis); marcadores diferentes (o, s, ^, D) como segunda
  codificação, então a figura se lê em tons de cinza.
- Linha de 1,5 pt, marcador com anel branco, grade hairline discreta, texto em cinza
  escuro (nunca na cor da série), legenda sempre presente (4 séries), IC de 95% como
  barra de erro quando maior que o marcador.
- Cada painel tem a própria escala vertical quando a teoria muda de painel.

Figuras:
 1. violacao_vs_n_{p90,p95,p99}: violação simulada sem contenção x 1 − q^N (linha).
    Uma única figura com os 3 limiares em painéis: `fig_violacao_vs_n`.
 2. fig_violacao_vs_c_{p90,p95,p99}: violação x C, um painel por N, ξ em séries,
    linha tracejada = teoria 1 − q^N.
 3. fig_amplificacao_vs_c: p99/p50 do conjunto x C, um painel por N, ξ em séries
    (escala logarítmica).
 4. fig_concorrencia_efetiva: concorrência efetiva x C, um painel por N; a diagonal
    pontilhada é "nenhum laço preso".

    python -m src.analysis.figuras --gerar
    python -m src.analysis.teste_figuras
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

from src.analysis.carga import RAIZ  # noqa: E402
from src.analysis.metricas import NOMES_LIMIAR  # noqa: E402
from src.analysis.teoria import CAMINHO_CENARIOS, Q_VIOLACAO, ler_tabela_cenarios, violacao_teorica  # noqa: E402

PASTA_DATA = RAIZ / "data"
PASTA_FIG = RAIZ / "results" / "figuras"

SUPERFICIE = "#ffffff"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
GRADE = "#e6e5e1"
REF = "#52514e"
# azul de uma matiz, claro -> escuro, ξ = 0,2; 0,5; 0,8; 1,1 (validado como rampa ordinal)
COR_XI = {0.2: "#86b6ef", 0.5: "#3987e5", 0.8: "#1c5cab", 1.1: "#0d366b"}
MARCA_XI = {0.2: "o", 0.5: "s", 0.8: "^", 1.1: "D"}


def _estilo() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
        "axes.facecolor": SUPERFICIE, "figure.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2, "text.color": TEXTO, "axes.titlecolor": TEXTO,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 8.5, "axes.titleweight": "regular", "legend.frameon": False,
    })


def _rotulo_xi(xi: float) -> str:
    return f"ξ = {str(xi).replace('.', ',')}"


def _ler_csv(caminho: Path) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _serie(idx: dict, chaves: list, col: str) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([idx[k][f"{col}_media"] for k in chaves], dtype=float),
            np.array([idx[k][f"{col}_ic95"] for k in chaves], dtype=float))


def _plotar_xi(ax, xs, medias, ics, xi, deslocamento=None):
    x = np.asarray(xs, dtype=float)
    if deslocamento is not None:
        x = x * deslocamento
    ax.errorbar(x, medias, yerr=ics, color=COR_XI[xi], lw=1.5, marker=MARCA_XI[xi], ms=5.0,
                mfc=COR_XI[xi], mec=SUPERFICIE, mew=1.0, ecolor=COR_XI[xi], elinewidth=0.8,
                capsize=0, zorder=3)


def _legenda_xi(ax_ou_fig, xis, extra=None, **kw):
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=COR_XI[x], lw=1.5, marker=MARCA_XI[x], ms=5.0, mfc=COR_XI[x],
                mec=SUPERFICIE, label=_rotulo_xi(x)) for x in xis]
    if extra:
        h += extra
    return ax_ou_fig.legend(handles=h, **kw)


def _salvar(fig, pasta: Path, nome: str) -> list[Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    saidas = []
    for ext in ("pdf", "png"):
        p = pasta / f"{nome}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        saidas.append(p)
    plt.close(fig)
    return saidas


def fig_violacao_vs_n(cenarios: list[dict], pasta: Path) -> list[Path]:
    cap = max(c["concorrencia"] for c in cenarios if c["sem_contencao"])
    idx = {(c["n_fan_out"], c["concorrencia"], c["xi"]): c for c in cenarios}
    ns = sorted({c["n_fan_out"] for c in cenarios})
    xis = sorted({c["xi"] for c in cenarios})
    fig, eixos = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
    n_fino = np.linspace(1, max(ns), 200)
    for ax, nome in zip(eixos, NOMES_LIMIAR):
        q = Q_VIOLACAO[nome]
        ax.plot(n_fino, 1 - q**n_fino, color=REF, lw=1.0, ls=(0, (4, 3)), zorder=2)
        for k, xi in enumerate(xis):
            m, e = _serie(idx, [(n, cap, xi) for n in ns], f"violacao_{nome}")
            _plotar_xi(ax, ns, m, e, xi, deslocamento=np.exp((k - 1.5) * 0.05))
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_title(f"limiar {nome} ($q$ = {str(q).replace('.', ',')})")
        ax.set_xlabel("fan-out $N$")
        ax.set_ylim(-0.02, 1.02)
    eixos[0].set_ylabel("fração de requisições acima do limiar")
    from matplotlib.lines import Line2D

    ref = Line2D([], [], color=REF, lw=1.0, ls=(0, (4, 3)), label="teoria $1-q^N$")
    _legenda_xi(fig, xis, extra=[ref], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))
    return _salvar(fig, pasta, "fig_violacao_vs_n")


def _painel_por_n(cenarios, pasta, nome, col, rotulo_y, teoria=None, logy=False, diagonal=False,
                  c_min=None):
    """Grade 2x3: um painel por N, ξ em séries, C no eixo x; o 6º painel guarda a legenda."""
    cap = max(c["concorrencia"] for c in cenarios if c["sem_contencao"])
    idx = {(c["n_fan_out"], c["concorrencia"], c["xi"]): c for c in cenarios}
    ns = sorted({c["n_fan_out"] for c in cenarios})
    xis = sorted({c["xi"] for c in cenarios})
    cs = sorted({c["concorrencia"] for c in cenarios if (c_min is None or c["concorrencia"] >= c_min)})
    fig, eixos = plt.subplots(2, 3, figsize=(6.3, 4.4))
    for ax, n in zip(eixos.flat, ns):
        for xi in xis:
            m, e = _serie(idx, [(n, c, xi) for c in cs], col)
            _plotar_xi(ax, cs, m, e, xi)
        if teoria is not None:
            ax.axhline(teoria(n), color=REF, lw=1.0, ls=(0, (4, 3)), zorder=2)
        if diagonal:
            ax.plot([cs[0], cs[-1]], [cs[0], cs[-1]], color=REF, lw=1.0, ls=(0, (1, 2)), zorder=2)
        if cap in cs and cs[0] < cap:
            ax.axvline(cap, color=GRADE, lw=0.9, zorder=1)
        if logy:
            ax.set_yscale("log")
            fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(cs)
        ax.set_title(f"$N$ = {n}")
        ax.set_xlabel("concorrência $C$")
        ax.set_ylabel(rotulo_y)
    for ax in eixos.flat[len(ns):]:
        ax.set_axis_off()
    from matplotlib.lines import Line2D

    extra = []
    if teoria is not None:
        extra.append(Line2D([], [], color=REF, lw=1.0, ls=(0, (4, 3)), label="teoria $1-q^N$"))
    if diagonal:
        extra.append(Line2D([], [], color=REF, lw=1.0, ls=(0, (1, 2)), label="nenhum laço preso"))
    if cap in cs and cs[0] < cap:
        extra.append(Line2D([], [], color=GRADE, lw=0.9, label=f"capacidade ($C$ = {cap})"))
    if len(ns) < 6:
        _legenda_xi(eixos.flat[5], xis, extra=extra, loc="center", fontsize=8)
    fig.tight_layout()
    return _salvar(fig, pasta, nome)


def fig_violacao_vs_c(cenarios: list[dict], pasta: Path, limiar: str) -> list[Path]:
    q = Q_VIOLACAO[limiar]
    return _painel_por_n(cenarios, pasta, f"fig_violacao_vs_c_{limiar}", f"violacao_{limiar}",
                         f"violação (limiar {limiar})", teoria=lambda n: violacao_teorica(n, q))


def fig_amplificacao_vs_c(cenarios: list[dict], pasta: Path) -> list[Path]:
    return _painel_por_n(cenarios, pasta, "fig_amplificacao_vs_c", "amplificacao_p99_p50",
                         "p99/p50 do conjunto", logy=True, c_min=4)


def fig_concorrencia_efetiva(linhas: list[dict], pasta: Path) -> list[Path]:
    cen = []
    for x in linhas:
        cen.append({"id": x["id"], "n_fan_out": int(x["n_fan_out"]), "concorrencia": int(x["concorrencia"]),
                    "xi": float(x["xi"]), "sem_contencao": x["sem_contencao"] == "true",
                    "c_efetivo_media": float(x["c_efetivo_media"]), "c_efetivo_ic95": float(x["c_efetivo_ic95"])})
    return _painel_por_n(cen, pasta, "fig_concorrencia_efetiva", "c_efetivo",
                         "concorrência efetiva", diagonal=True, c_min=4)


def gerar(pasta_data: Path = PASTA_DATA, pasta_saida: Path = PASTA_FIG) -> list[Path]:
    _estilo()
    cen = ler_tabela_cenarios(pasta_data / "metricas_cenarios.csv")
    saidas = fig_violacao_vs_n(cen, pasta_saida)
    for nome in NOMES_LIMIAR:
        saidas += fig_violacao_vs_c(cen, pasta_saida, nome)
    saidas += fig_amplificacao_vs_c(cen, pasta_saida)
    saidas += fig_concorrencia_efetiva(_ler_csv(pasta_data / "concorrencia_efetiva.csv"), pasta_saida)
    return saidas


def main() -> None:
    ap = argparse.ArgumentParser(description="Figuras do texto (Fase 6, item 4)")
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--data", default=str(PASTA_DATA))
    ap.add_argument("--saida", default=str(PASTA_FIG))
    a = ap.parse_args()
    if not a.gerar:
        ap.error("use --gerar")
    saidas = gerar(Path(a.data), Path(a.saida))
    for p in saidas:
        print(f"gravado: {p}")


if __name__ == "__main__":
    main()
