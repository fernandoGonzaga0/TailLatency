"""Modelagem dos recursos (APIs abstratas) — Etapa 5, passo 2.

Cada "API" é um simpy.Resource: um ou mais atendentes (capacity) com uma
fila FIFO ilimitada. Uma chamada que chega e encontra todos os atendentes
ocupados espera na fila; a latência observada é, portanto,

    latência = tempo de espera na fila + tempo de serviço

Decisões que este módulo implementa
-----------------------------------
- Número de APIs por cenário = grau de fan-out N (decidido em 21/10/2026).
  Cada requisição dispara uma chamada para cada uma das N APIs, como no
  modelo fork-join do Qiu et al. (2015). Assim cada API recebe exatamente
  uma chamada por requisição e, sem contenção, vale P(violação) = 1 - q^N.
- Capacidade por API configurável por cenário, padrão 1 (decidido em
  21/10/2026). A fila é FIFO e ilimitada (comportamento padrão do Resource).
- Todas as APIs de um cenário compartilham os mesmos parâmetros de latência
  (decisão de 06/10/2026); por isso a capacidade também é única por cenário.
- O limiar de SLA é calculado sobre o tempo de serviço isolado, sem fila
  (seção 2.2 do documento de contexto). Por isso o tempo de espera é
  devolvido separado do tempo de serviço.

Escopo
------
`chamar_api` recebe o tempo de serviço pronto. `chamar_api_sorteada` o sorteia
do gerador lognormal + GPD (`latencia.py`, passo 3). O disparo das N
chamadas (fan-out) e dos níveis de concorrência é o passo 4. O tempo simulado
é adimensional; por convenção, 1.0 = 1 ms.
"""

from dataclasses import dataclass

import numpy as np
import simpy

from src.simulation.latencia import ParamsLatencia


@dataclass(frozen=True)
class ResultadoChamada:
    """Medidas de uma única chamada a uma API."""

    api_id: int
    chegada: float  # instante simulado em que a chamada chegou
    espera: float   # tempo na fila (0 quando há atendente livre)
    servico: float  # tempo de serviço (o valor sorteado, sem fila)

    @property
    def latencia(self) -> float:
        """Latência individual observada: espera + serviço."""
        return self.espera + self.servico


def criar_apis(
    env: simpy.Environment, n_apis: int, capacidade: int = 1
) -> list[simpy.Resource]:
    """Cria as APIs de um cenário.

    n_apis: número de APIs (igual ao fan-out N do cenário).
    capacidade: chamadas que cada API atende ao mesmo tempo.
    """
    if n_apis < 1:
        raise ValueError("n_apis deve ser >= 1")
    if capacidade < 1:
        raise ValueError("capacidade deve ser >= 1")
    return [simpy.Resource(env, capacity=capacidade) for _ in range(n_apis)]


def chamar_api(
    env: simpy.Environment,
    api: simpy.Resource,
    api_id: int,
    tempo_servico: float,
):
    """Processo SimPy: uma chamada a uma API.

    Uso, dentro de outro processo:

        resultado = yield env.process(chamar_api(env, api, i, servico))

    ou, para disparar sem esperar, env.process(chamar_api(...)). O valor
    devolvido é um ResultadoChamada.
    """
    if tempo_servico < 0:
        raise ValueError("tempo_servico deve ser >= 0")
    chegada = env.now
    with api.request() as req:
        yield req  # espera na fila, se não houver atendente livre
        espera = env.now - chegada
        yield env.timeout(tempo_servico)
    return ResultadoChamada(
        api_id=api_id, chegada=chegada, espera=espera, servico=tempo_servico
    )


def chamar_api_sorteada(
    env: simpy.Environment,
    api: simpy.Resource,
    api_id: int,
    params: ParamsLatencia,
    rng: np.random.Generator,
):
    """Processo SimPy: uma chamada com tempo de serviço sorteado de `params`.

    O tempo de serviço é sorteado quando a chamada é disparada, antes de
    entrar na fila, então não depende de quanto ela vai esperar. O SimPy
    inicia os processos em ordem determinística; assim, a mesma seed no `rng`
    reproduz os mesmos tempos.

    Devolve um ResultadoChamada, como `chamar_api`.
    """
    tempo_servico = params.sortear(rng)
    resultado = yield from chamar_api(env, api, api_id, tempo_servico)
    return resultado
