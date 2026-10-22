"""Verificação do módulo de recursos — Etapa 5, passo 2.

Confere, com tempo de serviço fixo (1.0), que a fila do simpy.Resource se
comporta como esperado e, depois, a integração com o gerador de latência
(`chamar_api_sorteada`). Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_recursos

Se aparecer "Recursos funcionando corretamente.", o passo 2 está verificado.
"""

import numpy as np
import simpy

from src.simulation.latencia import ParamsLatencia
from src.simulation.recursos import (
    ResultadoChamada,
    chamar_api,
    chamar_api_sorteada,
    criar_apis,
)


def _rodar(n_chamadas: int, capacidade: int, servico: float = 1.0):
    """Dispara n_chamadas simultâneas a uma única API e devolve os resultados."""
    env = simpy.Environment()
    (api,) = criar_apis(env, n_apis=1, capacidade=capacidade)
    procs = [env.process(chamar_api(env, api, 0, servico)) for _ in range(n_chamadas)]
    env.run()
    return [p.value for p in procs]


def teste_fila_com_capacidade_1():
    r = _rodar(n_chamadas=3, capacidade=1)
    assert [x.latencia for x in r] == [1.0, 2.0, 3.0], r
    assert [x.espera for x in r] == [0.0, 1.0, 2.0], r
    assert all(x.servico == 1.0 for x in r)


def teste_sem_contencao_com_capacidade_suficiente():
    r = _rodar(n_chamadas=3, capacidade=3)
    assert [x.latencia for x in r] == [1.0, 1.0, 1.0], r
    assert all(x.espera == 0.0 for x in r)


def teste_capacidade_2_forma_fila_so_para_o_excedente():
    r = _rodar(n_chamadas=3, capacidade=2)
    assert sorted(x.latencia for x in r) == [1.0, 1.0, 2.0], r


def teste_uma_chamada_por_api_sem_contencao():
    """Fan-out N com N APIs e uma requisição: ninguém espera."""
    n = 5
    env = simpy.Environment()
    apis = criar_apis(env, n_apis=n, capacidade=1)
    procs = [env.process(chamar_api(env, a, i, 1.0)) for i, a in enumerate(apis)]
    env.run()
    r = [p.value for p in procs]
    assert len(apis) == n
    assert all(isinstance(x, ResultadoChamada) for x in r)
    assert all(x.espera == 0.0 and x.latencia == 1.0 for x in r)
    assert [x.api_id for x in r] == list(range(n))


def teste_validacao_de_argumentos():
    env = simpy.Environment()
    for kwargs in ({"n_apis": 0}, {"n_apis": 1, "capacidade": 0}):
        try:
            criar_apis(env, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"deveria falhar com {kwargs}")


def _rodar_sorteado(n_chamadas: int, capacidade: int, seed: int, n_apis: int = 1):
    """Dispara n_chamadas simultâneas (distribuídas em ordem entre as APIs)
    com tempo de serviço sorteado de ParamsLatencia()."""
    env = simpy.Environment()
    apis = criar_apis(env, n_apis=n_apis, capacidade=capacidade)
    params = ParamsLatencia()
    rng = np.random.default_rng(seed)
    procs = [
        env.process(chamar_api_sorteada(env, apis[k % n_apis], k % n_apis, params, rng))
        for k in range(n_chamadas)
    ]
    env.run()
    return [p.value for p in procs]


def teste_sorteada_sem_contencao_latencia_igual_ao_servico():
    n = 5
    r = _rodar_sorteado(n_chamadas=n, capacidade=1, seed=3, n_apis=n)
    esperado = ParamsLatencia().sortear(np.random.default_rng(3), n)  # mesma seed e ordem
    assert all(x.espera == 0.0 for x in r)
    assert np.allclose([x.servico for x in r], esperado)
    assert np.allclose([x.latencia for x in r], esperado)


def teste_sorteada_com_contencao_latencias_acumulam_na_fila():
    """Capacidade 1, FIFO: a latência da i-ésima chamada é a soma dos serviços até ela."""
    r = _rodar_sorteado(n_chamadas=4, capacidade=1, seed=5)
    servicos = [x.servico for x in r]
    assert np.allclose([x.latencia for x in r], np.cumsum(servicos))
    assert r[0].espera == 0.0 and all(x.espera > 0.0 for x in r[1:])


def teste_sorteada_reprodutivel_com_a_mesma_seed():
    a = [x.latencia for x in _rodar_sorteado(6, capacidade=2, seed=11, n_apis=2)]
    b = [x.latencia for x in _rodar_sorteado(6, capacidade=2, seed=11, n_apis=2)]
    c = [x.latencia for x in _rodar_sorteado(6, capacidade=2, seed=12, n_apis=2)]
    assert a == b and a != c


def main():
    testes = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"ok  {t.__name__}")
    print("\nRecursos funcionando corretamente.")


if __name__ == "__main__":
    main()
