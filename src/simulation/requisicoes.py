"""Processos de requisição e fan-out em sistema fechado — Etapa 5, passo 4.

Uma requisição dispara uma chamada a cada uma das N APIs ao mesmo tempo
(fan-out) e só termina quando a mais lenta responde. A latência do conjunto é,
portanto, o máximo das N latências individuais (espera na fila + serviço).

Decisão de modelagem: concorrência em sistema fechado (21/10/2026)
-----------------------------------------------------------------
O nível de concorrência C é o número fixo de requisições em andamento. Cada
um dos C "laços" executa uma requisição atrás da outra, sem tempo de reflexão
(think time zero): assim que uma requisição termina, o mesmo laço dispara a
seguinte. Justificativas:

- Combina com a hipótese ("o risco aumenta conforme o nível de concorrência"):
  o nível de concorrência é o próprio parâmetro C.
- Precedente: Tennage et al. (2019) simulam em SimPy um sistema fechado com
  fila FCFS e variam a concorrência; Harchol-Balter (2013, seção 2.6) descreve
  o sistema fechado com think time zero e reconhece que fechado versus aberto é
  uma escolha de modelagem (Schroeder et al., NSDI 2006).
- Um sistema aberto (chegadas com taxa lambda) não tem regime estacionário
  quando a GPD tem xi >= 1 (média de serviço infinita) e a espera média da
  fila diverge para xi >= 0,5 (variância infinita). O sistema fechado é sempre
  estável, pois C é limitado. (Argumento do TCC, não de uma fonte.)
- Limitação a registrar: a carga se autorregula, pois se as latências sobem as
  requisições passam a ser disparadas mais devagar. Sistemas abertos ou
  parcialmente abertos ficam como extensão.

Consequências para a contenção
------------------------------
Cada laço mantém, no máximo, uma chamada por API em andamento, então cada API
recebe no máximo C chamadas simultâneas. Sem contenção: C = 1, ou
capacidade >= C. Com C > capacidade as chamadas esperam na fila (recursos.py).

Início sincronizado
-------------------
No instante 0 os C laços disparam juntos, um transitório que não representa o
regime permanente. `n_aquecimento` descarta as primeiras requisições iniciadas.

O tempo simulado é adimensional; por convenção, 1.0 = 1 ms.

Como verificar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_requisicoes
"""

from dataclasses import dataclass

import numpy as np
import simpy

from src.simulation.latencia import ParamsLatencia
from src.simulation.recursos import ResultadoChamada, chamar_api, criar_apis


@dataclass(frozen=True)
class ResultadoRequisicao:
    """Medidas de uma requisição (fan-out de N chamadas)."""

    id_requisicao: int
    laco: int  # qual dos C laços concorrentes a executou (0 a C-1)
    inicio: float
    fim: float
    chamadas: tuple[ResultadoChamada, ...]  # uma por API, na ordem das APIs

    @property
    def latencia_conjunto(self) -> float:
        """Latência do conjunto: tempo até a chamada mais lenta responder."""
        return self.fim - self.inicio

    @property
    def servico_conjunto(self) -> float:
        """Máximo dos tempos de serviço, sem fila (conjunto sem contenção)."""
        return max(c.servico for c in self.chamadas)


def requisicao(
    env: simpy.Environment,
    apis: list[simpy.Resource],
    params: ParamsLatencia,
    rng: np.random.Generator,
    id_requisicao: int,
    laco: int,
):
    """Processo SimPy: dispara uma chamada a cada API e espera todas.

    Devolve um ResultadoRequisicao.
    """
    inicio = env.now
    # Os N tempos de serviço saem de uma única chamada vetorizada ao gerador
    # (muito mais rápido que N chamadas escalares ao scipy) e são sorteados no
    # disparo, antes de qualquer fila, como em chamar_api_sorteada.
    servicos = params.sortear(rng, len(apis))
    procs = [
        env.process(chamar_api(env, api, i, float(servicos[i])))
        for i, api in enumerate(apis)
    ]
    yield env.all_of(procs)
    return ResultadoRequisicao(
        id_requisicao=id_requisicao,
        laco=laco,
        inicio=inicio,
        fim=env.now,
        chamadas=tuple(p.value for p in procs),
    )


def _laco(env, laco, apis, params, rng, estado, total, saida):
    """Um dos C laços: requisições em sequência, sem think time."""
    while estado["proximo"] < total:
        id_req = estado["proximo"]
        estado["proximo"] += 1
        res = yield env.process(requisicao(env, apis, params, rng, id_req, laco))
        saida.append(res)


def rodar_cenario(
    n_fan_out: int,
    concorrencia: int,
    n_requisicoes: int,
    params: ParamsLatencia = ParamsLatencia(),
    capacidade: int = 1,
    seed: int = 2026,
    n_aquecimento: int = 0,
) -> list[ResultadoRequisicao]:
    """Roda um cenário em sistema fechado e devolve as requisições medidas.

    n_fan_out: N, número de APIs (e de chamadas por requisição).
    concorrencia: C, requisições em andamento ao mesmo tempo.
    n_requisicoes: quantas requisições medir (depois do aquecimento).
    capacidade: atendentes por API.
    seed: semente do gerador; a mesma seed reproduz o resultado.
    n_aquecimento: primeiras requisições iniciadas a descartar.

    A lista sai ordenada por id_requisicao, que segue a ordem de disparo.
    """
    if n_fan_out < 1:
        raise ValueError("n_fan_out deve ser >= 1")
    if concorrencia < 1:
        raise ValueError("concorrencia deve ser >= 1")
    if n_requisicoes < 1:
        raise ValueError("n_requisicoes deve ser >= 1")
    if n_aquecimento < 0:
        raise ValueError("n_aquecimento deve ser >= 0")

    env = simpy.Environment()
    apis = criar_apis(env, n_fan_out, capacidade)
    rng = np.random.default_rng(seed)
    total = n_aquecimento + n_requisicoes
    estado = {"proximo": 0}
    saida: list[ResultadoRequisicao] = []
    for laco in range(concorrencia):
        env.process(_laco(env, laco, apis, params, rng, estado, total, saida))
    env.run()
    saida.sort(key=lambda r: r.id_requisicao)
    return [r for r in saida if r.id_requisicao >= n_aquecimento]
