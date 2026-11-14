"""Verificação da comparação com a teoria — Fase 6, item 3.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.teste_teoria

Se aparecer "Teoria funcionando corretamente.", as fórmulas, a classificação e o
critério estão verificados. Roda em segundos e não lê dados reais.
"""

import math
import tempfile
from pathlib import Path

import numpy as np

from src.analysis import teoria as T
from src.analysis.agregacao import COLUNAS_CENARIOS, gravar_csv
from src.analysis.metricas import METRICAS

GERADOR = {"mediana_ms": 10.0, "sigma": 0.5, "p_corte": 0.95}


def teste_violacao_formula():
    assert abs(T.violacao_teorica(1, 0.95) - 0.05) < 1e-15
    assert abs(T.violacao_teorica(10, 0.99) - (1 - 0.99**10)) < 1e-15
    assert T.violacao_teorica(50, 0.95) > 0.92
    for ruim in [(0, 0.9), (5, 0.0), (5, 1.0)]:
        try:
            T.violacao_teorica(*ruim)
        except ValueError:
            continue
        raise AssertionError("deveria recusar " + str(ruim))


def teste_quantil_conjunto():
    p = T.params_do_gerador(GERADOR, 0.5)
    # N = 1: o conjunto é a própria chamada
    for q in (0.5, 0.95, 0.99):
        assert abs(T.quantil_conjunto(p, 1, q) - p.quantil(q)) < 1e-12
    # a cdf do máximo no quantil teórico devolve p
    for n in (5, 20, 50):
        for q in (0.5, 0.95, 0.99):
            x = T.quantil_conjunto(p, n, q)
            assert abs(float(p.cdf(x)) ** n - q) < 1e-9
    # cresce com N
    assert T.quantil_conjunto(p, 50, 0.5) > T.quantil_conjunto(p, 5, 0.5)
    # o limiar p95 individual tem violação teórica 1-0.95^N pelo quantil do conjunto
    n = 10
    x = p.quantil(0.95)
    assert abs((1 - float(p.cdf(x)) ** n) - T.violacao_teorica(n, 0.95)) < 1e-9
    assert T.amplificacao_teorica(p, 1) > 1.0


def teste_monte_carlo_contra_formulas():
    """Máximo de N sorteios i.i.d. do gerador reproduz violação e quantis teóricos."""
    rng = np.random.default_rng(7)
    p = T.params_do_gerador(GERADOR, 0.5)
    n, reps = 10, 200_000
    maximos = p.sortear(rng, (reps, n)).max(axis=1)
    for nome, q in T.Q_VIOLACAO.items():
        limiar = p.quantil(q)
        frac = float(np.mean(maximos > limiar))
        teoria = T.violacao_teorica(n, q)
        ep = math.sqrt(teoria * (1 - teoria) / reps)
        assert abs(frac - teoria) < 5 * ep + 1e-9, (nome, frac, teoria)
    for nome, q in (("p50", 0.5), ("p95", 0.95)):
        emp = float(np.percentile(maximos, 100 * q))
        teo = T.quantil_conjunto(p, n, q)
        assert abs(emp / teo - 1) < 0.02, (nome, emp, teo)


def teste_posicao():
    assert T.posicao(0.5, 0.52, 0.03) == "dentro"
    assert T.posicao(0.5, 0.5299, 0.03) == "dentro"  # quase na borda
    assert T.posicao(0.5, 0.5301, 0.03) == "acima"
    assert T.posicao(0.5, 0.60, 0.03) == "acima"
    assert T.posicao(0.5, 0.40, 0.03) == "abaixo"
    assert T.posicao(0.5, 0.5, 0.0) == "dentro"


def _cenario(id_, n, c, xi, sem, desloc=0.0, ic=0.01):
    """Linha de metricas_cenarios coerente com a teoria (+ deslocamento na violação)."""
    p = T.params_do_gerador(GERADOR, xi)
    linha = {"id": id_, "n_fan_out": n, "concorrencia": c, "xi": xi,
             "sem_contencao": sem, "n_replicacoes": 10}
    for m in METRICAS:
        linha[f"{m}_media"] = 1.0
        linha[f"{m}_ic95"] = ic
    for m in T.METRICAS:
        linha[f"{m}_media"] = T.teoria_da_metrica(m, p, n) + (desloc if m.startswith("violacao_p") else 0.0)
        linha[f"{m}_ic95"] = ic
    return linha


def _grade_sintetica(desloc_sem=0.0, desloc_com=0.0):
    cs = []
    for xi in (0.2, 0.5):
        for n in (1, 5, 10):
            cs.append(_cenario(f"s{n}_{xi}", n, 1, xi, True, desloc_sem))
            cs.append(_cenario(f"c{n}_{xi}", n, 8, xi, False, desloc_com))
    return cs


def teste_comparar_e_resumo():
    comp = T.comparar(_grade_sintetica(), GERADOR)
    assert len(comp) == 12
    assert [x["id"] for x in comp] == sorted(x["id"] for x in comp)
    for x in comp:
        for m in T.METRICAS:
            assert x[f"{m}_posicao"] == "dentro" and abs(x[f"{m}_dif"]) < 1e-12
    r = T.resumir(comp)
    assert r["criterio"]["testes"] == 18 and r["criterio"]["fora"] == 0 and r["criterio"]["aprovado"]
    assert r["sem_contencao"]["violacao_p95"]["total"] == 6
    assert r["com_contencao"]["violacao_p95"]["total"] == 6

    # simulado bem acima da teoria sem contenção: todos fora, reprova
    comp = T.comparar(_grade_sintetica(desloc_sem=0.2), GERADOR)
    r = T.resumir(comp)
    assert r["criterio"]["fora"] == 18 and not r["criterio"]["aprovado"]
    assert all(x["violacao_p95_posicao"] == "acima" for x in comp if x["sem_contencao"])
    # o desvio com contenção NÃO entra no critério
    comp = T.comparar(_grade_sintetica(desloc_com=0.2), GERADOR)
    r = T.resumir(comp)
    assert r["criterio"]["fora"] == 0 and r["criterio"]["aprovado"]
    assert r["com_contencao"]["violacao_p95"]["acima"] == 6
    # abaixo
    comp = T.comparar(_grade_sintetica(desloc_sem=-0.2), GERADOR)
    assert T.resumir(comp)["sem_contencao"]["violacao_p99"]["abaixo"] == 6


def teste_limite_do_criterio():
    base = T.comparar(_grade_sintetica(), GERADOR)
    # 10 fora: aprova; 11 fora: reprova (limite pelo parâmetro)
    for k, esperado in ((10, True), (11, False)):
        comp = [dict(x) for x in base]
        marcados = 0
        for x in comp:
            if not x["sem_contencao"]:
                continue
            for m in T.METRICAS_VIOLACAO:
                if marcados < k:
                    x[f"{m}_posicao"] = "acima"
                    marcados += 1
        assert T.resumir(comp, limite_fora=10)["criterio"]["aprovado"] is esperado


def teste_leitura_e_gravacao():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cs = _grade_sintetica()
        # tabela de entrada no formato real (com as demais colunas)
        gravar_csv(cs, COLUNAS_CENARIOS, d / "cenarios.csv")
        lidas = T.ler_tabela_cenarios(d / "cenarios.csv")
        assert len(lidas) == 12 and isinstance(lidas[0]["sem_contencao"], bool)
        assert isinstance(lidas[0]["n_fan_out"], int)
        resumo = T.gerar(d / "cenarios.csv", d / "saida.csv", GERADOR)
        assert resumo["criterio"]["aprovado"]
        b1 = (d / "saida.csv").read_bytes()
        T.gerar(d / "cenarios.csv", d / "saida.csv", GERADOR)
        assert b1 == (d / "saida.csv").read_bytes()  # determinístico
        assert b1.decode().splitlines()[0].split(",") == list(T.COLUNAS)
        assert len(b1.decode().splitlines()) == 13
        # coluna faltando
        texto = (d / "cenarios.csv").read_text(encoding="utf-8")
        cab, *resto = texto.splitlines()
        novo = cab.replace("violacao_p95_ic95,", "")
        (d / "ruim.csv").write_text("\n".join([novo, *resto]), encoding="utf-8")
        try:
            T.ler_tabela_cenarios(d / "ruim.csv")
        except ValueError:
            pass
        else:
            raise AssertionError("deveria recusar tabela sem coluna")


def teste_gerador_da_grade():
    g = T.carregar_gerador(T.CAMINHO_GRADE)
    assert g == GERADOR, g
    p = T.params_do_gerador(g, 0.5)
    assert abs(p.u - 22.7602) < 1e-3 and abs(p.beta - 5.5170) < 1e-3


def main():
    for f in (teste_violacao_formula, teste_quantil_conjunto, teste_monte_carlo_contra_formulas,
              teste_posicao, teste_comparar_e_resumo, teste_limite_do_criterio,
              teste_leitura_e_gravacao, teste_gerador_da_grade):
        f()
        print(f"  ok: {f.__name__}")
    print("Teoria funcionando corretamente.")


if __name__ == "__main__":
    main()
