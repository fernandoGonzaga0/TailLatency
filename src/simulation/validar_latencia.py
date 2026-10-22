"""Validação visual e numérica do gerador de latência — Etapa 5, passo 3.

Gera, para cada nível de xi, o histograma (corpo) com a densidade teórica, o
QQ-plot e a curva de sobrevivência (cauda) em escala log-log, e imprime uma
tabela de quantis teóricos contra empíricos e o teste de Kolmogorov-Smirnov.

Como rodar (venv ativado, a partir da raiz do projeto TCC1):

    python -m src.simulation.validar_latencia

A figura é salva em results/validacao_latencia.png (a pasta results/ é
ignorada pelo git). Opções: --n (tamanho da amostra por xi), --seed e
--saida (caminho da figura).
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from src.simulation.latencia import ParamsLatencia

XIS = (0.2, 0.5, 0.8, 1.1)
PS = (0.5, 0.9, 0.95, 0.99, 0.999)
RAIZ = Path(__file__).resolve().parents[2]


def tabela(n: int, seed: int) -> dict:
    """Imprime a comparação numérica e devolve as amostras por xi."""
    amostras = {}
    print(f"Amostra de {n:,} sorteios por xi (seed {seed}); tempos em ms\n".replace(",", "."))
    for xi in XIS:
        p = ParamsLatencia(xi=xi)
        x = p.sortear(np.random.default_rng(seed), n)
        amostras[xi] = x
        ks = kstest(x, p.cdf)
        print(f"xi = {xi}  (u = {p.u:.2f} ms, beta = {p.beta:.3f} ms)  "
              f"KS = {ks.statistic:.5f}, p-valor = {ks.pvalue:.3f}  "
              f"fração > u = {np.mean(x > p.u):.4f}")
        print("   p        teórico   empírico   erro relativo")
        for prob in PS:
            teo = p.quantil(prob)
            emp = float(np.quantile(x, prob))
            print(f"   {prob:<7}  {teo:8.2f}  {emp:9.2f}  {100 * (emp - teo) / teo:+9.2f}%")
        print()
    return amostras


def figura(amostras: dict, saida: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(3, len(XIS), figsize=(4.2 * len(XIS), 11))
    for j, xi in enumerate(XIS):
        p = ParamsLatencia(xi=xi)
        x = amostras[xi]
        n = len(x)

        # 1) histograma do corpo (até o p99 teórico) com a densidade teórica
        ax = eixos[0, j]
        xmax = p.quantil(0.99)
        bins = np.linspace(0, xmax, 81)
        largura = bins[1] - bins[0]
        ax.hist(x, bins=bins, weights=np.full(n, 1 / (n * largura)),
                color="#9ecae1", label="amostra")
        grade = np.linspace(1e-3, xmax, 600)
        ax.plot(grade, p.pdf(grade), color="#08519c", lw=1.5, label="densidade teórica")
        ax.axvline(p.u, color="gray", ls="--", lw=1, label="corte u")
        ax.set_title(f"xi = {xi}")
        ax.set_xlabel("latência (ms), até o p99")
        ax.set_ylabel("densidade")
        if j == 0:
            ax.legend(fontsize=8)

        # 2) QQ-plot em log-log
        ax = eixos[1, j]
        probs = np.concatenate([np.linspace(0.001, 0.9, 200), 1 - np.logspace(-1, -4, 200)])
        teo = p.quantil(probs)
        emp = np.quantile(x, probs)
        ax.loglog(teo, emp, ".", ms=3, color="#08519c")
        lim = [teo.min(), teo.max()]
        ax.loglog(lim, lim, color="gray", lw=1)
        ax.set_xlabel("quantil teórico (ms)")
        ax.set_ylabel("quantil empírico (ms)")
        ax.set_title("QQ-plot (até p99,99)")

        # 3) curva de sobrevivência (cauda) em log-log
        ax = eixos[2, j]
        xs = np.sort(x)
        pontos = np.logspace(np.log10(p.quantil(0.5)), np.log10(p.quantil(1 - 1e-4)), 200)
        emp_s = 1 - np.searchsorted(xs, pontos, side="right") / n
        ax.loglog(pontos, 1 - p.cdf(pontos), color="#08519c", lw=1.5, label="teórica")
        ax.loglog(pontos, np.where(emp_s > 0, emp_s, np.nan), ".", ms=3,
                  color="#de2d26", label="empírica")
        ax.axvline(p.u, color="gray", ls="--", lw=1)
        ax.set_xlabel("latência (ms)")
        ax.set_ylabel("P(L > x)")
        ax.set_title("Sobrevivência (cauda)")
        if j == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Validação do gerador lognormal + GPD", y=0.995)
    fig.tight_layout()
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=130)
    print(f"Figura salva em {saida}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--saida", type=Path, default=RAIZ / "results" / "validacao_latencia.png")
    args = ap.parse_args()
    figura(tabela(args.n, args.seed), args.saida)


if __name__ == "__main__":
    main()
