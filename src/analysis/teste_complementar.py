"""Verificação das análises complementares — Fase 6, item 7.

    python -m src.analysis.teste_complementar

Se aparecer "Complementar funcionando corretamente.", as medidas de excesso, as
médias sobre ξ e o efeito de ξ estão verificados (dados sintéticos).
"""

import math
import tempfile
from pathlib import Path

from src.analysis import complementar as K
from src.analysis.metricas import METRICAS
from src.analysis.teoria import Q_VIOLACAO, violacao_teorica


def _grade(excesso=0.1, ic=0.01):
    """N em 1,10; C em 4 (sem contenção) e 8 (com); ξ em 0.2, 0.5, 1.1.
    O excesso do cenário com contenção é `excesso` menos 0,02 por degrau de ξ."""
    cs = []
    for n in (1, 10):
        for c in (4, 8):
            for k, xi in enumerate((0.2, 0.5, 1.1)):
                sem = c <= 4
                linha = {"id": f"N{n}_C{c}_xi{xi}", "n_fan_out": n, "concorrencia": c,
                         "xi": xi, "sem_contencao": sem, "n_replicacoes": 10}
                for m in METRICAS:
                    linha[f"{m}_media"], linha[f"{m}_ic95"] = 1.0, ic
                for nome, q in Q_VIOLACAO.items():
                    t = violacao_teorica(n, q)
                    ex = 0.0 if sem else min(excesso - 0.02 * k, 0.99 - t)
                    linha[f"violacao_{nome}_media"] = t + ex
                    linha[f"violacao_{nome}_ic95"] = ic
                cs.append(linha)
    return cs


def teste_excesso_por_cenario():
    exc = K.excesso_por_cenario(_grade())
    assert len(exc) == 2 * 3 * 3  # 2 N x 3 ξ x 3 limiares, só com contenção
    assert all(x["concorrencia"] == 8 for x in exc)
    x = next(e for e in exc if e["n_fan_out"] == 10 and e["limiar"] == "p95" and e["xi"] == 0.2)
    t = violacao_teorica(10, 0.95)
    assert abs(x["teoria"] - t) < 1e-15 and abs(x["dif"] - min(0.1, 0.99 - t)) < 1e-12
    assert abs(x["razao"] - x["simulado"] / t) < 1e-12
    assert abs(x["razao_ic95"] - 0.01 / t) < 1e-12
    assert abs(x["folga"] - (1 - t)) < 1e-12
    assert abs(x["frac_folga"] - x["dif"] / (1 - t)) < 1e-12
    assert abs(x["dif_ic95"] - 0.01) < 1e-15


def teste_media_sobre_xi():
    exc = K.excesso_por_cenario(_grade())
    nc = K.excesso_por_n_c(exc)
    assert len(nc) == 2 * 3  # 2 N x 3 limiares (C=8 apenas)
    g = [e for e in exc if e["n_fan_out"] == 1 and e["limiar"] == "p90"]
    m = next(e for e in nc if e["n_fan_out"] == 1 and e["limiar"] == "p90")
    assert abs(m["dif"] - sum(e["dif"] for e in g) / 3) < 1e-12
    assert abs(m["dif_ic95"] - math.sqrt(3 * 0.01**2) / 3) < 1e-12
    assert abs(m["razao"] - m["simulado"] / m["teoria"]) < 1e-12


def teste_efeito_xi():
    exc = K.excesso_por_cenario(_grade(excesso=0.2, ic=0.001))
    ef = K.efeito_xi(exc)
    assert len(ef) == 2 * 2 * 3  # 2 N x 2 ξ (sem a base) x 3 limiares
    assert all(x["xi_base"] == 0.2 for x in ef)
    x = next(e for e in ef if e["n_fan_out"] == 1 and e["limiar"] == "p90" and e["xi"] == 1.1)
    assert abs(x["dif_xi"] - (-0.04)) < 1e-12  # 0,2 − 0,04 contra 0,2
    assert abs(x["dif_xi_ic95"] - math.hypot(0.001, 0.001)) < 1e-15
    assert x["significativo"] is True
    # ruído grande: diferença deixa de ser significativa
    exc = K.excesso_por_cenario(_grade(excesso=0.2, ic=0.05))
    assert not any(e["significativo"] for e in K.efeito_xi(exc))
    try:
        K.efeito_xi(exc, xi_base=0.9)
    except ValueError:
        pass
    else:
        raise AssertionError("deveria recusar xi_base ausente")


def teste_gravacao():
    from src.analysis.agregacao import COLUNAS_CENARIOS, gravar_csv

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        gravar_csv(_grade(), COLUNAS_CENARIOS, d / "c.csv")
        K.gerar(d / "c.csv", d)
        antes = {n: (d / n).read_bytes() for n in
                 ("excesso_violacao.csv", "excesso_por_n_c.csv", "efeito_xi.csv")}
        K.gerar(d / "c.csv", d)
        assert antes == {n: (d / n).read_bytes() for n in antes}
        cab = antes["excesso_violacao.csv"].decode().splitlines()
        assert cab[0].split(",") == list(K.COLUNAS_EXCESSO) and len(cab) == 19
        assert antes["efeito_xi.csv"].decode().splitlines()[0].split(",") == list(K.COLUNAS_XI)


def main():
    for f in (teste_excesso_por_cenario, teste_media_sobre_xi, teste_efeito_xi, teste_gravacao):
        f()
        print(f"  ok: {f.__name__}")
    print("Complementar funcionando corretamente.")


if __name__ == "__main__":
    main()
