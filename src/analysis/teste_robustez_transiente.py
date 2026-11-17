"""Verificação da robustez ao transiente — Fase 6, verificação A.

    python -m src.analysis.teste_robustez_transiente

Se aparecer "Robustez ao transiente funcionando corretamente.", as medidas, a
agregação e o veredito estão verificados (dados sintéticos e uma simulação pequena).
"""

import pandas as pd

from src.analysis import robustez_transiente as R
from src.simulation.requisicoes import rodar_cenario


def _tabela(lat):
    return pd.DataFrame({"id_requisicao": range(len(lat)), "latencia_conjunto_ms": lat})


def teste_fim_conhecido():
    # 1000 requisições; limiar 10; só as últimas 500 violam em 20% (100 de 500)
    lat = [5.0] * 500 + ([20.0] * 100 + [5.0] * 400)
    m = R.fim_execucao_execucao(_tabela(lat), 10.0)
    assert m["viol_resto"] == 0.0 and abs(m["viol_fim"] - 0.2) < 1e-12 and m["n_total"] == 1000
    assert abs(m["viol_total"] - 0.1) < 1e-12
    # igual ao limiar não viola; ordena por id mesmo com linhas embaralhadas
    t = _tabela([10.0] * 600).sample(frac=1.0, random_state=1)
    assert R.fim_execucao_execucao(t, 10.0)["viol_total"] == 0.0
    t = _tabela([1.0] * 100 + [50.0] * 5).sample(frac=1.0, random_state=2)
    assert R.fim_execucao_execucao(t, 10.0, janela=5)["viol_fim"] == 1.0
    for args in ((_tabela([1.0] * 5), 10.0), (_tabela([1.0] * 600), 10.0, 0)):
        try:
            R.fim_execucao_execucao(*args)
        except ValueError:
            continue
        raise AssertionError("deveria recusar")


def teste_sobreposicao():
    assert R.difere_sobreposicao(0.5, 0.01, 0.6, 0.01) is True
    assert R.difere_sobreposicao(0.5, 0.06, 0.6, 0.05) is False
    assert R.difere_sobreposicao(0.5, 0.05, 0.6, 0.05) is False  # na fronteira: não difere
    assert R.violacao([1, 2, 3, 4], 2) == 0.5


def teste_agregar_fim_e_veredito():
    execs = []
    for rep in range(10):
        for id_, delta in (("A", 0.0), ("B", 0.3)):
            ruido = 0.001 * (rep % 3)
            execs.append({"id": id_, "n_fan_out": 5, "concorrencia": 8, "xi": 0.8, "replicacao": rep,
                          "viol_fim": 0.5 + delta + ruido, "viol_resto": 0.5 + ruido,
                          "viol_total": 0.5, "n_total": 1000})
    a, b = R.agregar_fim(execs)
    assert a["id"] == "A" and a["n_replicacoes"] == 10
    assert a["difere_sobreposicao"] is False and a["difere_pareada"] is False
    assert b["difere_sobreposicao"] is True and b["difere_pareada"] is True
    # veredito: 9 de 10 passam = 0,9 -> aprovado; 8 de 10 -> reprovado
    ok = R._linha_veredito("x", "criterio", [False] * 9 + [True])
    assert ok["passam"] == 9 and ok["aprovado"] is True
    ruim = R._linha_veredito("x", "criterio", [False] * 8 + [True] * 2)
    assert ruim["aprovado"] is False
    assert R._linha_veredito("x", "criterio", [])["aprovado"] is False


def teste_agregar_aquecimento():
    execs = []
    for rep in range(10):
        r = 0.001 * (rep % 3)
        for w, base in ((0, 0.40), (500, 0.50), (2000, 0.50)):
            execs.append({"id": "A", "n_fan_out": 5, "concorrencia": 8, "xi": 0.8, "replicacao": rep,
                          "aquecimento": w, "viol": base + r, "n": 100})
    (linha,) = R.agregar_aquecimento(execs)
    assert linha["n_replicacoes"] == 10
    assert abs(linha["viol_w500_media"] - 0.5009) < 1e-12 or abs(linha["viol_w500_media"] - 0.5 - 0.0009) < 1e-9
    assert linha["difere_sobreposicao_2000_500"] is False and linha["difere_pareada_2000_500"] is False
    assert linha["difere_sobreposicao_500_0"] is True and linha["difere_pareada_500_0"] is True
    assert abs(linha["dif_500_0_media"] - 0.10) < 1e-12
    execs.pop()  # réplica faltando em um aquecimento
    try:
        R.agregar_aquecimento(execs)
    except ValueError:
        pass
    else:
        raise AssertionError("deveria recusar réplicas diferentes")


def teste_conferir_reproducao():
    fim = [{"id": "A", "replicacao": 0, "viol_total": 0.5}]
    ok = [{"id": "A", "replicacao": 0, "aquecimento": 500, "viol": 0.5001},
          {"id": "A", "replicacao": 0, "aquecimento": 2000, "viol": 0.9}]
    assert R.conferir_reproducao(ok, fim, 10000) == 1
    ruim = [{"id": "A", "replicacao": 0, "aquecimento": 500, "viol": 0.52}]
    try:
        R.conferir_reproducao(ruim, fim, 10000)
    except ValueError:
        pass
    else:
        raise AssertionError("deveria recusar divergência")


def teste_simulacao_pequena():
    # mesma seed, mesmo nº de medidas; aquecimento diferente muda quais ids entram,
    # e aquecimento w reproduz a cauda da execução com aquecimento menor
    kw = dict(n_fan_out=2, concorrencia=3, n_requisicoes=50, capacidade=2, seed=7)
    a = rodar_cenario(n_aquecimento=0, **kw)
    b = rodar_cenario(n_aquecimento=20, **kw)
    assert len(a) == len(b) == 50 and b[0].id_requisicao == 20
    c = rodar_cenario(n_aquecimento=0, **{**kw, "n_requisicoes": 70})
    assert [r.latencia_conjunto for r in b] == [r.latencia_conjunto for r in c[20:]]


def main():
    for f in (teste_fim_conhecido, teste_sobreposicao, teste_agregar_fim_e_veredito,
              teste_agregar_aquecimento, teste_conferir_reproducao, teste_simulacao_pequena):
        f()
        print(f"  ok: {f.__name__}")
    print("Robustez ao transiente funcionando corretamente.")


if __name__ == "__main__":
    main()
