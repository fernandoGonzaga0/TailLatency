"""Verificação da concorrência efetiva — Fase 6, item 9.

    python -m src.analysis.teste_concorrencia_efetiva

Se aparecer "Concorrência efetiva funcionando corretamente.", a medida e a
agregação estão verificadas (execuções sintéticas, sem ler o bruto real).
"""

import math

import pandas as pd

from src.analysis import concorrencia_efetiva as K


def _tabela(intervalos):
    ini = [a for a, _ in intervalos]
    fim = [b for _, b in intervalos]
    lat = [b - a for a, b in intervalos]
    return pd.DataFrame({"inicio_ms": ini, "fim_ms": fim, "latencia_conjunto_ms": lat})


def teste_medida_conhecida():
    # C = 2: um laço preso 100 ms; o outro faz 10 requisições de 10 ms; limiar 1, K = 10
    t = _tabela([(0, 100)] + [(10 * i, 10 * i + 10) for i in range(10)])
    m = K.medidas_execucao(t, 2, 1.0)
    assert abs(m["f_preso"] - 0.5) < 1e-12 and abs(m["c_efetivo"] - 1.0) < 1e-12
    assert abs(m["frac_req_presas"] - 1 / 11) < 1e-12 and m["tempo_total_ms"] == 100
    # a comparação é estrita: latência exatamente K x limiar não é presa
    m = K.medidas_execucao(_tabela([(0, 10), (10, 20)]), 1, 1.0)
    assert m["f_preso"] == 0.0 and m["c_efetivo"] == 1.0
    # sem requisições presas a concorrência efetiva é C
    assert K.medidas_execucao(_tabela([(0, 5), (5, 9)]), 4, 1.0)["c_efetivo"] == 4.0


def teste_recorte_e_validacao():
    # duração fora de [menor início, maior fim] não existe por construção; mas f <= 1
    t = _tabela([(0, 50), (0, 50)])
    m = K.medidas_execucao(t, 2, 1.0)
    assert abs(m["f_preso"] - 1.0) < 1e-12 and abs(m["c_efetivo"]) < 1e-12
    for args in ((_tabela([(0, 1)]), 0, 1.0), (_tabela([(0, 1)]), 1, 0.0),
                 (_tabela([(5, 5)]), 1, 1.0)):
        try:
            K.medidas_execucao(*args)
        except ValueError:
            continue
        raise AssertionError("deveria recusar " + str(args[1:]))


def teste_agregar():
    execs = []
    for rep in range(10):
        for id_, c in (("A", 8), ("B", 2)):
            execs.append({"id": id_, "n_fan_out": 5, "concorrencia": c, "xi": 0.5, "replicacao": rep,
                          "f_preso": 0.1 * (rep % 2), "c_efetivo": c * (1 - 0.1 * (rep % 2)),
                          "frac_req_presas": 0.0, "tempo_total_ms": 1000.0})
    tab = K.agregar(execs, capacidade=4)
    assert [x["id"] for x in tab] == ["A", "B"]
    a, b = tab
    assert a["sem_contencao"] is False and b["sem_contencao"] is True and a["n_replicacoes"] == 10
    assert abs(a["f_preso_media"] - 0.05) < 1e-12
    # IC t: t(9) x desvio-padrão amostral / raiz(10)
    dp = math.sqrt(sum((0.1 * (r % 2) - 0.05) ** 2 for r in range(10)) / 9)
    assert abs(a["f_preso_ic95"] - 2.2621571627982053 * dp / math.sqrt(10)) < 1e-9
    assert a["tempo_total_ms_ic95"] == 0.0 or abs(a["tempo_total_ms_ic95"]) < 1e-12

def teste_direcao():
    ic = [0.001] * 4
    assert K.direcao([0.1, 0.2, 0.3, 0.4], ic) == "cresce"
    assert K.direcao([0.4, 0.3, 0.2, 0.1], ic) == "decresce"
    assert K.direcao([0.2, 0.2, 0.2, 0.2], ic) == "nenhuma"
    assert K.direcao([0.0, 0.0], [0.0, 0.0]) == "nenhuma"  # tudo zero (sem requisições presas)
    # não monotônica com queda significativa: nem cresce nem decresce
    assert K.direcao([0.1, 0.4, 0.1, 0.4], ic) == "nenhuma"


def _tabela_k(f_por_k):
    """Tabelas sintéticas por K: 1 N, 3 valores de C, 2 de ξ; f_preso = f(k, c, xi)."""
    tabelas = {}
    for k, f in f_por_k.items():
        tabelas[k] = [
            {"id": f"N005_C{c:02d}_xi{xi:.2f}", "n_fan_out": 5, "concorrencia": c, "xi": xi,
             "f_preso_media": f(c, xi), "f_preso_ic95": 0.001}
            for c in (1, 4, 8) for xi in (0.2, 0.5)
        ]
    return tabelas


def teste_series_e_veredito_k():
    cresce_c = lambda c, xi: 0.01 * c + 0.1 * xi
    tabelas = _tabela_k({5.0: cresce_c, 10.0: cresce_c, 20.0: cresce_c})
    d = K.series_direcao(tabelas)
    assert len(d) == 2 + 3  # por ξ (série em C) e por C (série em ξ)
    assert all(x["igual_k5"] and x["igual_k20"] for x in d)
    assert {x["dir_k10"] for x in d if x["efeito"] == "C"} == {"cresce"}
    assert all(v["aprovado"] and v["fracao"] == 1.0 for v in K.veredito_k(d))
    # K = 20 inverte a tendência em C: 2 de 5 séries mudam -> 60% < 90%
    tabelas = _tabela_k({5.0: cresce_c, 10.0: cresce_c, 20.0: lambda c, xi: -0.01 * c + 0.1 * xi})
    v = {x["k"]: x for x in K.veredito_k(K.series_direcao(tabelas))}
    assert v[5]["aprovado"] is True and v[20]["aprovado"] is False
    assert v[20]["passam"] == 3 and v[20]["testes"] == 5


def teste_conferir_k10(tmp_path=None):
    import tempfile
    from pathlib import Path

    linhas = [{"id": "A", **{f"{m}_{s}": 0.5 for m in K.MEDIDAS for s in ("media", "ic95")}}]
    with tempfile.TemporaryDirectory() as d:
        caminho = Path(d) / "x.csv"
        caminho.write_text(
            "id," + ",".join(f"{m}_{s}" for m in K.MEDIDAS for s in ("media", "ic95")) + "\n"
            + "A," + ",".join("0.5" for _ in range(2 * len(K.MEDIDAS))) + "\n", encoding="utf-8")
        assert K.conferir_k10(linhas, caminho) == 1
        linhas[0]["f_preso_media"] = 0.6
        for ruim in (linhas, [{"id": "B", **linhas[0]}]):
            try:
                K.conferir_k10(ruim, caminho)
            except ValueError:
                continue
            raise AssertionError("deveria recusar divergência")


def main():
    for f in (teste_medida_conhecida, teste_recorte_e_validacao, teste_agregar,
              teste_direcao, teste_series_e_veredito_k, teste_conferir_k10):
        f()
        print(f"  ok: {f.__name__}")
    print("Concorrência efetiva funcionando corretamente.")


if __name__ == "__main__":
    main()
