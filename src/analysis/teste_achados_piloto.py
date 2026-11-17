"""Verificação dos achados do piloto — Fase 6, item 9.

    python -m src.analysis.teste_achados_piloto

Se aparecer "Achados do piloto funcionando corretamente.", as tabelas estão
verificadas (dados sintéticos).
"""

import tempfile
from pathlib import Path

from src.analysis import achados_piloto as A
from src.analysis.metricas import METRICAS


def _grade(cai=True):
    cs = []
    for n in (1, 50):
        for c in (4, 6, 8):
            sem = c <= 4
            linha = {"id": f"N{n}_C{c}", "n_fan_out": n, "concorrencia": c, "xi": 0.5,
                     "sem_contencao": sem, "n_replicacoes": 10}
            for m in METRICAS:
                linha[f"{m}_media"], linha[f"{m}_ic95"] = 1.0, 0.01
            amp = 4.0 - 0.5 * (c - 4) / 2 if cai else 4.0 + 0.5 * (c - 4) / 2
            linha["amplificacao_p99_p50_media"] = amp
            linha["p50_ms_media"] = 10.0 * (1 + 0.3 * (c - 4) / 2)
            linha["p99_ms_media"] = 40.0 * (1 + 0.1 * (c - 4) / 2) if cai else 40.0 * (1 + 0.8 * (c - 4) / 2)
            cs.append(linha)
    return cs


def teste_amplificacao():
    r = A.amplificacao_vs_c(_grade(cai=True))
    assert len(r) == 2 and all(x["decresce"] for x in r)
    x = r[0]
    assert (x["c_inicial"], x["c_final"]) == (4, 8)
    assert abs(x["amp_inicial"] - 4.0) < 1e-12 and abs(x["amp_final"] - 3.0) < 1e-12
    assert abs(x["razao_amp"] - 0.75) < 1e-12
    assert abs(x["p50_razao"] - 1.6) < 1e-12 and abs(x["p99_razao"] - 1.2) < 1e-12
    assert x["mediana_sobe_mais"] is True
    r = A.amplificacao_vs_c(_grade(cai=False))
    assert not any(x["decresce"] for x in r) and not any(x["mediana_sobe_mais"] for x in r)


def teste_teto():
    t = A.tabela_teto(_grade())
    assert len(t) == 2 * 3
    d = {(x["n_fan_out"], x["limiar"]): x for x in t}
    assert d[(50, "p90")]["teto"] and d[(50, "p95")]["teto"] is False  # 0,9231 < 0,95
    assert not d[(1, "p90")]["teto"]
    assert abs(d[(50, "p90")]["folga"] + d[(50, "p90")]["teoria"] - 1.0) < 1e-15


def teste_gravacao():
    from src.analysis.agregacao import COLUNAS_CENARIOS, gravar_csv

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        gravar_csv(_grade(), COLUNAS_CENARIOS, d / "c.csv")
        A.gerar(d / "c.csv", d)
        a1, b1 = (d / "amplificacao_vs_c.csv").read_bytes(), (d / "teto.csv").read_bytes()
        A.gerar(d / "c.csv", d)
        assert a1 == (d / "amplificacao_vs_c.csv").read_bytes() and b1 == (d / "teto.csv").read_bytes()
        assert a1.decode().splitlines()[0].split(",") == list(A.COLUNAS_AMP)
        assert b1.decode().splitlines()[0].split(",") == list(A.COLUNAS_TETO)


def main():
    for f in (teste_amplificacao, teste_teto, teste_gravacao):
        f()
        print(f"  ok: {f.__name__}")
    print("Achados do piloto funcionando corretamente.")


if __name__ == "__main__":
    main()
