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


def main():
    for f in (teste_medida_conhecida, teste_recorte_e_validacao, teste_agregar):
        f()
        print(f"  ok: {f.__name__}")
    print("Concorrência efetiva funcionando corretamente.")


if __name__ == "__main__":
    main()
