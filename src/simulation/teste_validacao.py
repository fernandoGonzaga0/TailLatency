"""Verificação da validação do módulo — Etapa 5, passo 8.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_validacao

Se aparecer "Validação funcionando corretamente.", as ferramentas do passo 8
estão verificadas (a validação em si roda com `python -m src.simulation.validacao --rodar`).
Usa uma grade pequena (2.000 requisições por replicação), então roda em segundos.
"""

import tempfile
from pathlib import Path

from src.simulation import validacao as v
from src.simulation.cenarios import buscar_cenario, carregar_grade

GRADE = """
[geral]
seed_base = 2026
capacidade = 4
n_requisicoes = 2000
n_aquecimento = 100
n_replicacoes = 2
[latencia]
mediana_ms = 10.0
sigma = 0.5
p_corte = 0.95
[grade]
fan_out = [1, 3]
concorrencia = [1, 6]
xi = [0.5]
"""


def _cenario(id_cenario="N003_C01_xi0.50"):
    p = Path(tempfile.mkdtemp()) / "g.toml"
    p.write_text(GRADE, encoding="utf-8")
    return buscar_cenario(carregar_grade(p), id_cenario)


def teste_wilson_valores_conhecidos():
    lo, hi = v.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = v.wilson(0, 100)
    assert lo < 1e-9 and 0 < hi < 0.05
    lo, hi = v.wilson(100, 100)
    assert hi > 1 - 1e-9 and 0.95 < lo < 1
    assert v.wilson(500, 1000)[0] > v.wilson(50, 100)[0]  # mais dados, IC mais estreito


def teste_teoria_de_violacao():
    assert abs(v.teoria_violacao(0.95, 10) - 0.4013) < 1e-4
    assert abs(v.teoria_violacao(0.99, 10) - 0.0956) < 1e-4
    assert abs(v.teoria_violacao(0.90, 10) - 0.6513) < 1e-4
    assert v.teoria_violacao(0.95, 1) == 1 - 0.95


def teste_quantil_do_conjunto():
    c1, c4 = _cenario("N001_C01_xi0.50"), _cenario("N003_C01_xi0.50")
    assert abs(v.quantil_conjunto(c1, 0.95) - c1.params.quantil(0.95)) < 1e-9
    for p in (0.5, 0.95, 0.99):
        q = v.quantil_conjunto(c4, p)
        assert abs(float(c4.params.cdf(q)) ** 3 - p) < 1e-9  # F(x)^N = p
    assert v.quantil_conjunto(c4, 0.5) > c4.params.quantil(0.5)  # máximo é maior


def teste_cenario_sem_contencao_e_aprovado_e_sem_espera():
    r = v.validar_cenario(_cenario())
    assert r["sem_espera"] and r["aprovado"]
    assert len(r["testes"]) == 9 and r["n_requisicoes"] == 4000
    assert {t["teste"] for t in r["testes"]} >= {"violacao_p90", "violacao_p95", "violacao_p99"}
    assert r["amplificacao_p99_p50"]["teorico"] > 1


def teste_reprodutivel():
    a, b = v.validar_cenario(_cenario()), v.validar_cenario(_cenario())
    assert a == b


def teste_cenario_com_contencao_e_recusado():
    try:
        v.validar_cenario(_cenario("N003_C06_xi0.50"))
    except ValueError:
        return
    raise AssertionError("cenário com contenção deveria ser recusado")


def teste_teoria_errada_e_detectada():
    c = _cenario()
    big = v.contar_replicacao(c, 0)
    n = big["n"]
    k = big["violacoes"]["0.95"]
    certo = v._teste("x", k, n, v.teoria_violacao(0.95, 3), "")
    errado = v._teste("x", k, n, v.teoria_violacao(0.90, 3), "")  # teoria de outro q
    assert certo["passou"] and not errado["passou"]


def teste_repeticao_so_dos_que_falham_e_sem_mudar_a_tolerancia():
    c = _cenario()
    original = v.agregar
    chamadas = []

    def falso(cenario, contagens):
        r = original(cenario, contagens)
        chamadas.append(cenario.seed_base)
        if len(chamadas) == 1:  # na primeira execução, um teste falha
            r["testes"][0]["passou"] = False
        return r

    v.agregar = falso
    try:
        r = v.validar_cenario(c)
    finally:
        v.agregar = original
    assert chamadas == [2026, 2026 + v.SEED_BASE_REPETICAO]  # outra seed_base, uma vez
    assert r["repetidos"] == ["violacao_p90"]
    t = r["testes"][0]
    assert not t["passou"] and t["aprovado_na_repeticao"] and t["aprovado"] and r["aprovado"]
    assert all("repeticao" not in x for x in r["testes"][1:])


def teste_falha_nas_duas_e_reprovada():
    c = _cenario()
    original = v.agregar

    def sempre_falha(cenario, contagens):
        r = original(cenario, contagens)
        r["testes"][1]["passou"] = False
        return r

    v.agregar = sempre_falha
    try:
        r = v.validar_cenario(c)
    finally:
        v.agregar = original
    assert not r["aprovado"]
    assert v.resumo_geral([r])["reprovados_apos_repeticao"] == 1


def teste_resumo_geral():
    r = v.validar_cenario(_cenario())
    s = v.resumo_geral([r])
    assert s["cenarios"] == 1 and s["testes"] == 9 and s["aprovado"]
    assert s["cenarios_sem_espera"] == 1 and abs(s["esperado_por_acaso"] - 0.05) < 1e-12


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Validação funcionando corretamente.")
