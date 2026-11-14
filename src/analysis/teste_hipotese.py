"""Verificação do módulo da hipótese — Fase 6, item 6.

    python -m src.analysis.teste_hipotese

Se aparecer "Hipótese funcionando corretamente.", a regra de crescimento, as
séries e o veredito estão verificados (dados sintéticos, roda em segundos).
"""

import tempfile
from pathlib import Path

from src.analysis import hipotese as H
from src.analysis.metricas import METRICAS
from src.analysis.teoria import Q_VIOLACAO, violacao_teorica


def teste_regra_cresce():
    r = H.cresce([0.1, 0.2, 0.3], [0.01, 0.01, 0.01])
    assert r == {"quedas": 0, "ganho": True, "cresce": True}
    # queda pequena dentro do ruído é tolerada
    assert H.cresce([0.1, 0.205, 0.2, 0.3], [0.01] * 4)["cresce"]
    # queda significativa reprova
    r = H.cresce([0.1, 0.3, 0.2, 0.4], [0.01] * 4)
    assert r["quedas"] == 1 and not r["cresce"]
    # sem ganho total significativo reprova (série plana)
    r = H.cresce([0.5, 0.5, 0.5], [0.01] * 3)
    assert r["quedas"] == 0 and not r["ganho"] and not r["cresce"]
    # queda exatamente no limite (diferença == soma dos IC) não conta como queda
    assert H.cresce([0.3, 0.1, 0.5], [0.1, 0.1, 0.01])["quedas"] == 0
    for ruim in ([0.1], [0.1, 0.2]):
        try:
            H.cresce(ruim, [0.01])
        except ValueError:
            continue
        raise AssertionError("deveria recusar série inválida")


def _grade(acima=0.1, cresce_c=True, ic=0.005):
    """Cenários sintéticos: N em 1,5,10; C em 1,4 (sem contenção) e 6,8 (com); ξ = 0.5."""
    cs = []
    for n in (1, 5, 10):
        for c in (1, 4, 6, 8):
            sem = c <= 4
            linha = {"id": f"N{n}_C{c}", "n_fan_out": n, "concorrencia": c, "xi": 0.5,
                     "sem_contencao": sem, "n_replicacoes": 10}
            for m in METRICAS:
                linha[f"{m}_media"], linha[f"{m}_ic95"] = 1.0, ic
            for nome, q in Q_VIOLACAO.items():
                t = violacao_teorica(n, q)
                extra = 0.0 if sem else acima * (c - 4) / 4 * (1 if cresce_c else -1)
                linha[f"violacao_{nome}_media"] = min(1.0, t + extra) if cresce_c else t + (acima if c == 6 else 0)
                linha[f"violacao_{nome}_ic95"] = ic
            cs.append(linha)
    return cs


def teste_series_e_veredito_aprovado():
    cs = _grade()
    em_n = H.series_em_n(cs)
    assert len(em_n) == 3 * 1 * 2  # 3 limiares x 1 ξ x C em {1,4}
    assert all(x["cresce"] for x in em_n)
    em_c = H.series_em_c(cs)
    assert len(em_c) == 3 * 3 * 1  # 3 limiares x 3 N x 1 ξ
    assert all(x["valores_x"] == "4 6 8" for x in em_c)
    res = H.avaliar(cs)
    assert len(res["veredito"]) == 9
    assert all(v["aprovado"] for v in res["veredito"]), res["veredito"]
    assert H.falhas(res) == {"2a": [], "series": []}


def teste_acima_e_queda_reprovam():
    # cenários com contenção sem excesso: nenhum 'acima'
    res = H.avaliar(_grade(acima=0.0))
    v = {(x["criterio"], x["limiar"]): x for x in res["veredito"]}
    assert not v[("2a_acima_da_teoria", "p95")]["aprovado"]
    assert v[("2a_acima_da_teoria", "p95")]["passam"] == 0
    # excesso que cai com C: série não cresce
    res = H.avaliar(_grade(acima=0.1, cresce_c=False))
    v = {(x["criterio"], x["limiar"]): x for x in res["veredito"]}
    assert not v[("2b_cresce_com_C", "p90")]["aprovado"]
    assert len(H.falhas(res)["series"]) > 0


def teste_gravacao():
    from src.analysis.agregacao import COLUNAS_CENARIOS, gravar_csv

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        gravar_csv(_grade(), COLUNAS_CENARIOS, d / "c.csv")
        r1 = H.gerar(d / "c.csv", d / "a.csv", d / "b.csv")
        a1, b1 = (d / "a.csv").read_bytes(), (d / "b.csv").read_bytes()
        H.gerar(d / "c.csv", d / "a.csv", d / "b.csv")
        assert a1 == (d / "a.csv").read_bytes() and b1 == (d / "b.csv").read_bytes()
        assert a1.decode().splitlines()[0].split(",") == list(H.COLUNAS_CRESCIMENTO)
        assert b1.decode().splitlines()[0].split(",") == list(H.COLUNAS_VEREDITO)
        assert len(b1.decode().splitlines()) == 10 and len(r1["series"]) == 15


def main():
    for f in (teste_regra_cresce, teste_series_e_veredito_aprovado,
              teste_acima_e_queda_reprovam, teste_gravacao):
        f()
        print(f"  ok: {f.__name__}")
    print("Hipótese funcionando corretamente.")


if __name__ == "__main__":
    main()
