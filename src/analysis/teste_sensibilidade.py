"""Verificação da sensibilidade ao limiar — Fase 6.

    python -m src.analysis.teste_sensibilidade

Se aparecer "Sensibilidade funcionando corretamente.", a classificação de direção,
a detecção de conflito e a correlação de postos estão verificadas (dados sintéticos).
"""

import tempfile
from pathlib import Path

from src.analysis import sensibilidade as S
from src.analysis.metricas import METRICAS


def _grade(modo="coerente"):
    """N em 1,10; C em 4 (sem contenção) e 8 (com); ξ em 0.2, 1.1."""
    cs = []
    for n in (1, 10):
        for c in (4, 8):
            for xi in (0.2, 1.1):
                linha = {"id": f"N{n}_C{c}_xi{xi}", "n_fan_out": n, "concorrencia": c, "xi": xi,
                         "sem_contencao": c <= 4, "n_replicacoes": 10}
                for m in METRICAS:
                    linha[f"{m}_media"], linha[f"{m}_ic95"] = 1.0, 0.01
                base = 0.1 * (1 if n == 1 else 5) + (0.2 if c == 8 else 0.0)
                for k, nome in enumerate(("p90", "p95", "p99")):
                    v = base * (1 - 0.1 * k)
                    if modo == "conflito" and nome == "p99" and xi == 1.1:
                        v -= 0.2  # limiar p99 vai no sentido oposto ao de p90 e p95 em ξ
                    if modo == "conflito" and nome != "p99" and xi == 1.1:
                        v += 0.2
                    if modo == "inverso" and nome == "p99":
                        v = 1.0 - v
                    linha[f"violacao_{nome}_media"] = v
                    linha[f"violacao_{nome}_ic95"] = 0.01
                cs.append(linha)
    return cs


def teste_direcao():
    assert S.direcao(0.1, 0.01, 0.5, 0.01) == "mais"
    assert S.direcao(0.5, 0.01, 0.1, 0.01) == "menos"
    assert S.direcao(0.5, 0.01, 0.51, 0.01) == "nulo"       # dentro do IC combinado
    assert S.direcao(0.5, 0.01, 0.5 + 0.0141, 0.01) == "nulo"  # no limite (hipot ~ 0,014142)
    assert S.direcao(0.5, 0.01, 0.5 + 0.0150, 0.01) == "mais"


def teste_tabela_direcao_coerente():
    d = S.tabela_direcao(_grade())
    assert [sum(1 for x in d if x["efeito"] == e) for e in ("N", "C", "xi")] == [2, 4, 4]
    assert not any(x["conflito"] for x in d)
    efeito_n = [x for x in d if x["efeito"] == "N"]
    assert all(x["dir_p90"] == x["dir_p95"] == x["dir_p99"] == "mais" for x in efeito_n)


def teste_conflito_detectado():
    d = S.tabela_direcao(_grade("conflito"))
    c = [x for x in d if x["conflito"]]
    assert c and all(x["efeito"] == "xi" for x in c)
    assert all(x["dir_p90"] == "mais" and x["dir_p99"] == "menos" for x in c)
    assert not S.avaliar(_grade("conflito"))["direcao_ok"]


def teste_ordem():
    o = S.tabela_ordem(_grade())
    assert {x["escopo"] for x in o} == {"global", "N=1", "N=10"} and len(o) == 9
    assert all(x["par"] in ("p90-p95", "p90-p99", "p95-p99") for x in o)
    assert all(abs(x["rho"] - 1.0) < 1e-12 for x in o)  # transformações monótonas
    assert S.avaliar(_grade())["ordem_ok"]
    inv = S.tabela_ordem(_grade("inverso"))
    assert any(x["rho"] < 0 and not x["aprovado"] for x in inv)
    assert not S.avaliar(_grade("inverso"))["ordem_ok"]
    # o mínimo é parâmetro
    assert all(x["aprovado"] for x in S.tabela_ordem(_grade("inverso"), minimo=-1.0))


def teste_gravacao():
    from src.analysis.agregacao import COLUNAS_CENARIOS, gravar_csv

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        gravar_csv(_grade(), COLUNAS_CENARIOS, d / "c.csv")
        S.gerar(d / "c.csv", d)
        a = (d / "sensibilidade_direcao.csv").read_bytes()
        b = (d / "sensibilidade_ordem.csv").read_bytes()
        S.gerar(d / "c.csv", d)
        assert a == (d / "sensibilidade_direcao.csv").read_bytes()
        assert b == (d / "sensibilidade_ordem.csv").read_bytes()
        assert a.decode().splitlines()[0].split(",") == list(S.COLUNAS_DIRECAO)
        assert b.decode().splitlines()[0].split(",") == list(S.COLUNAS_ORDEM)


def main():
    for f in (teste_direcao, teste_tabela_direcao_coerente, teste_conflito_detectado,
              teste_ordem, teste_gravacao):
        f()
        print(f"  ok: {f.__name__}")
    print("Sensibilidade funcionando corretamente.")


if __name__ == "__main__":
    main()
