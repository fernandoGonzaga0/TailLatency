"""Verificação da parametrização por cenário — Etapa 5, passo 5.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_cenarios

Se aparecer "Cenários funcionando corretamente.", o passo 5 está verificado.
"""

import math
import tempfile
from pathlib import Path

from src.simulation.cenarios import (
    CAMINHO_PADRAO,
    buscar_cenario,
    carregar_grade,
    rodar_replicacao,
)
from src.simulation.requisicoes import rodar_cenario

# seed de N010_C06_xi0.50, replicação 0, com seed_base 2026 (SeedSequence é estável)
_SEED_REFERENCIA = 3249627009

BASE = """
[geral]
seed_base = 2026
capacidade = 4
n_requisicoes = 300
n_aquecimento = 20
n_replicacoes = 3
[latencia]
mediana_ms = 10.0
sigma = 0.5
p_corte = 0.95
[grade]
fan_out = [1, 3]
concorrencia = [1, 4, 6]
xi = [0.2, 0.5]
"""


def _toml_temporario(texto: str) -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "g.toml"
    p.write_text(texto, encoding="utf-8")
    return p


def _deve_falhar(texto: str):
    try:
        carregar_grade(_toml_temporario(texto))
    except ValueError:
        return
    raise AssertionError(f"deveria falhar:\n{texto}")


def teste_grade_oficial_tem_120_cenarios():
    cs = carregar_grade(CAMINHO_PADRAO)
    assert len(cs) == 120
    assert {c.n_fan_out for c in cs} == {1, 5, 10, 20, 50}
    assert {c.concorrencia for c in cs} == {1, 4, 6, 8, 10, 12}
    assert {c.xi for c in cs} == {0.2, 0.5, 0.8, 1.1}
    assert all(c.capacidade == 4 for c in cs)
    assert all(c.n_requisicoes == 10000 and c.n_aquecimento == 500 for c in cs)
    assert all(c.n_replicacoes == 5 and c.seed_base == 2026 for c in cs)
    assert sum(c.n_replicacoes for c in cs) == 600


def teste_ids_unicos_e_formato():
    cs = carregar_grade(CAMINHO_PADRAO)
    ids = [c.id for c in cs]
    assert len(set(ids)) == 120
    assert buscar_cenario(cs, "N010_C06_xi0.50").n_fan_out == 10
    assert buscar_cenario(cs, "N050_C12_xi1.10").xi == 1.1
    try:
        buscar_cenario(cs, "N999_C99_xi9.99")
    except KeyError:
        pass
    else:
        raise AssertionError("id inexistente deveria falhar")


def teste_contencao_apenas_acima_da_capacidade():
    cs = carregar_grade(CAMINHO_PADRAO)
    assert all(c.sem_contencao == (c.concorrencia <= 4) for c in cs)
    assert sum(c.sem_contencao for c in cs) == 2 * 5 * 4  # C = 1 e C = 4


def teste_parametros_de_latencia():
    c = buscar_cenario(carregar_grade(CAMINHO_PADRAO), "N005_C01_xi0.80")
    p = c.params
    assert abs(p.mu - math.log(10.0)) < 1e-12
    assert p.sigma == 0.5 and p.p_corte == 0.95 and p.xi == 0.8
    assert abs(p.quantil(0.5) - 10.0) < 1e-6  # mediana de 10 ms
    assert abs(p.u - 22.7602) < 1e-3  # corte no p95 (passo 3)


def teste_seeds_deterministicas_e_distintas():
    cs = carregar_grade(CAMINHO_PADRAO)
    seeds = [c.seed(k) for c in cs for k in range(c.n_replicacoes)]
    assert len(set(seeds)) == 600  # nenhuma colisão
    assert cs[0].seed(0) == carregar_grade(CAMINHO_PADRAO)[0].seed(0)
    # valor de referência: se mudar, a reprodutibilidade entre versões quebrou
    assert buscar_cenario(cs, "N010_C06_xi0.50").seed(0) == _SEED_REFERENCIA


def teste_seed_nao_depende_da_posicao_na_grade():
    completa = carregar_grade(_toml_temporario(BASE))
    menor = carregar_grade(
        _toml_temporario(BASE.replace("[1, 4, 6]", "[6]").replace("[1, 3]", "[3]"))
    )
    assert len(completa) == 12 and len(menor) == 2
    a = buscar_cenario(completa, "N003_C06_xi0.50")
    b = buscar_cenario(menor, "N003_C06_xi0.50")
    assert all(a.seed(k) == b.seed(k) for k in range(3))


def teste_replicacao_fora_do_intervalo():
    c = carregar_grade(_toml_temporario(BASE))[0]
    for k in (-1, 3):
        try:
            c.seed(k)
        except ValueError:
            continue
        raise AssertionError(f"replicacao {k} deveria falhar")


def teste_rodar_replicacao_usa_a_seed_derivada_e_e_reprodutivel():
    c = buscar_cenario(carregar_grade(_toml_temporario(BASE)), "N003_C06_xi0.50")
    r1 = rodar_replicacao(c, 1)
    r2 = rodar_replicacao(c, 1)
    r3 = rodar_replicacao(c, 2)
    ref = rodar_cenario(
        n_fan_out=3,
        concorrencia=6,
        n_requisicoes=300,
        params=c.params,
        capacidade=4,
        seed=c.seed(1),
        n_aquecimento=20,
    )
    lat = lambda rs: [x.latencia_conjunto for x in rs]  # noqa: E731
    assert len(r1) == 300  # só as medidas; o aquecimento é descartado
    assert lat(r1) == lat(r2) == lat(ref)
    assert lat(r1) != lat(r3)


def teste_sem_contencao_nao_ha_espera_e_com_contencao_ha():
    cs = carregar_grade(_toml_temporario(BASE))
    sem = rodar_replicacao(buscar_cenario(cs, "N003_C04_xi0.50"), 0)
    com = rodar_replicacao(buscar_cenario(cs, "N003_C06_xi0.50"), 0)
    assert all(ch.espera == 0.0 for r in sem for ch in r.chamadas)
    assert any(ch.espera > 0.0 for r in com for ch in r.chamadas)


def teste_validacao_do_toml():
    _deve_falhar(BASE.replace("capacidade = 4", "capacidade = 0"))
    _deve_falhar(BASE.replace("n_replicacoes = 3", "n_replicacoes = 2.5"))
    _deve_falhar(BASE.replace("n_requisicoes = 300", "n_requisicoes = true"))
    _deve_falhar(BASE.replace("p_corte = 0.95", "p_corte = 1.0"))
    _deve_falhar(BASE.replace("sigma = 0.5", "sigma = -1.0"))
    _deve_falhar(BASE.replace("[1, 3]", "[]"))
    _deve_falhar(BASE.replace("[1, 3]", "[3, 3]"))
    _deve_falhar(BASE.replace("[1, 4, 6]", "[0, 4]"))
    _deve_falhar(BASE.replace("[0.2, 0.5]", "[0.255]"))
    _deve_falhar(BASE.replace("[0.2, 0.5]", "[0.0]"))
    _deve_falhar(BASE.replace("n_aquecimento = 20", "n_aquecimento = 20\nextra = 1"))  # chave desconhecida
    _deve_falhar(BASE.replace("seed_base = 2026\n", ""))  # chave faltando
    _deve_falhar(BASE + "\n[outra]\na = 1\n")  # seção desconhecida


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Cenários funcionando corretamente.")
