"""Verificação dos processos de requisição e fan-out — Etapa 5, passo 4.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_requisicoes

Se aparecer "Requisições funcionando corretamente.", o passo 4 está verificado.
Seed fixa 2026; a tolerância dos testes estatísticos é de 3,29 desvios-padrão
binomiais (99,9%) e não deve ser afrouxada.
"""

import numpy as np

from src.simulation.latencia import ParamsLatencia
from src.simulation.requisicoes import rodar_cenario

PARAMS = ParamsLatencia(xi=0.5)
SEED = 2026


def _fracao_acima(valores, limiar):
    return float(np.mean(np.asarray(valores) > limiar))


def teste_estrutura_da_requisicao():
    r = rodar_cenario(n_fan_out=4, concorrencia=1, n_requisicoes=50, params=PARAMS)
    assert len(r) == 50
    assert [x.id_requisicao for x in r] == list(range(50))
    for x in r:
        assert len(x.chamadas) == 4
        assert [c.api_id for c in x.chamadas] == [0, 1, 2, 3]
        # o conjunto é o máximo das latências individuais
        assert abs(x.latencia_conjunto - max(c.latencia for c in x.chamadas)) < 1e-9


def teste_fan_out_1_conjunto_igual_a_chamada_unica():
    r = rodar_cenario(n_fan_out=1, concorrencia=1, n_requisicoes=100, params=PARAMS)
    for x in r:
        assert abs(x.latencia_conjunto - x.chamadas[0].latencia) < 1e-9


def teste_sem_contencao_concorrencia_1_nao_ha_espera():
    r = rodar_cenario(n_fan_out=5, concorrencia=1, n_requisicoes=500, params=PARAMS)
    assert all(c.espera == 0.0 for x in r for c in x.chamadas)
    assert all(abs(x.latencia_conjunto - x.servico_conjunto) < 1e-9 for x in r)


def teste_sem_contencao_capacidade_igual_a_concorrencia():
    r = rodar_cenario(
        n_fan_out=5, concorrencia=4, capacidade=4, n_requisicoes=500, params=PARAMS
    )
    assert all(c.espera == 0.0 for x in r for c in x.chamadas)


def teste_com_contencao_ha_espera():
    r = rodar_cenario(
        n_fan_out=5, concorrencia=4, capacidade=1, n_requisicoes=500, params=PARAMS
    )
    assert any(c.espera > 0.0 for x in r for c in x.chamadas)
    assert all(x.latencia_conjunto >= x.servico_conjunto - 1e-9 for x in r)


def teste_sistema_fechado_c_requisicoes_em_andamento():
    # cada laço executa requisições em sequência, sem sobreposição nem folga
    c = 4
    r = rodar_cenario(
        n_fan_out=3, concorrencia=c, capacidade=1, n_requisicoes=400, params=PARAMS
    )
    por_laco = {}
    for x in r:
        por_laco.setdefault(x.laco, []).append(x)
    assert set(por_laco) == set(range(c))
    for reqs in por_laco.values():
        reqs.sort(key=lambda x: x.inicio)
        for a, b in zip(reqs, reqs[1:]):
            assert abs(b.inicio - a.fim) < 1e-9  # think time zero


def teste_fila_respeita_a_capacidade_de_cada_api():
    for capacidade in (1, 2):
        r = rodar_cenario(
            n_fan_out=3,
            concorrencia=5,
            capacidade=capacidade,
            n_requisicoes=300,
            params=PARAMS,
        )
        for api in range(3):
            eventos = []
            for x in r:
                c = x.chamadas[api]
                ini = c.chegada + c.espera
                eventos += [(ini, 1), (ini + c.servico, -1)]
            eventos.sort(key=lambda e: (e[0], e[1]))  # fim antes do início no empate
            ativos = 0
            for _, delta in eventos:
                ativos += delta
                assert ativos <= capacidade, (api, capacidade)


def teste_sem_contencao_bate_com_1_menos_q_elevado_a_n():
    # P(conjunto > p95 individual) = 1 - 0,95^N (hipótese, primeira parte)
    n_fan, n_req = 10, 20_000
    r = rodar_cenario(
        n_fan_out=n_fan, concorrencia=1, n_requisicoes=n_req, params=PARAMS, seed=SEED
    )
    p = 1 - 0.95**n_fan
    f = _fracao_acima([x.latencia_conjunto for x in r], PARAMS.quantil(0.95))
    dp = (p * (1 - p) / n_req) ** 0.5
    assert abs(f - p) <= 3.29 * dp, (f, p)


def teste_contencao_aumenta_a_violacao_com_a_concorrencia():
    limiar = PARAMS.quantil(0.95)
    fr = []
    for c in (1, 4, 8):
        r = rodar_cenario(
            n_fan_out=5,
            concorrencia=c,
            capacidade=1,
            n_requisicoes=5_000,
            params=PARAMS,
            seed=SEED,
            n_aquecimento=100,
        )
        fr.append(_fracao_acima([x.latencia_conjunto for x in r], limiar))
    assert fr[0] < fr[1] < fr[2], fr
    assert fr[0] < 1 - 0.95**5 + 0.03  # C = 1 fica na teoria
    assert fr[1] > 1 - 0.95**5  # com contenção, acima da teoria


def teste_aquecimento_descarta_as_primeiras():
    r = rodar_cenario(
        n_fan_out=2, concorrencia=3, n_requisicoes=40, params=PARAMS, n_aquecimento=10
    )
    assert len(r) == 40
    assert r[0].id_requisicao == 10


def teste_reprodutivel_com_a_mesma_seed():
    def lat(seed):
        r = rodar_cenario(
            n_fan_out=4,
            concorrencia=3,
            capacidade=1,
            n_requisicoes=200,
            params=PARAMS,
            seed=seed,
        )
        return [x.latencia_conjunto for x in r]

    assert lat(7) == lat(7)
    assert lat(7) != lat(8)


def teste_validacao_de_argumentos():
    for kwargs in (
        dict(n_fan_out=0, concorrencia=1, n_requisicoes=1),
        dict(n_fan_out=1, concorrencia=0, n_requisicoes=1),
        dict(n_fan_out=1, concorrencia=1, n_requisicoes=0),
        dict(n_fan_out=1, concorrencia=1, n_requisicoes=1, n_aquecimento=-1),
        dict(n_fan_out=1, concorrencia=1, n_requisicoes=1, capacidade=0),
    ):
        try:
            rodar_cenario(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"deveria falhar: {kwargs}")


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Requisições funcionando corretamente.")
