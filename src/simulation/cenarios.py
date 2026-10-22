"""Parametrização por cenário — Etapa 5, passo 5.

Lê a grade de cenários de `config/grade.toml` (TOML, lido pelo `tomllib` da
biblioteca padrão: sem dependência nova e com comentários no arquivo) e a
expande em cenários, cada um executável isoladamente e de forma repetível.

Grade decidida em 22/10/2026 (piloto com capacity = 4):

- fan-out N em {1, 5, 10, 20, 50}; concorrência C em {1, 4, 6, 8, 10, 12};
  peso da cauda xi em {0,2; 0,5; 0,8; 1,1}: 5 x 6 x 4 = 120 cenários;
- capacity = 4 fixa; C <= 4 não tem contenção, C > 4 tem fila;
- 5 replicações independentes por cenário, de 10.000 requisições medidas, com
  500 requisições de aquecimento descartadas.

Seeds
-----
A seed de cada replicação vem de `numpy.random.SeedSequence` com a entropia
[seed_base, capacidade, N, C, round(100 * xi), replicação]. Ela depende só do
cenário e da replicação, não da posição na grade: acrescentar ou retirar níveis
da grade não muda a seed dos cenários que ficam. Replicações e cenários
diferentes recebem seeds diferentes (amostras independentes).

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.cenarios --listar
    python -m src.simulation.cenarios --rodar N010_C06_xi0.50 --replicacao 0
    python -m src.simulation.teste_cenarios
"""

import argparse
import math
import tomllib
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from src.simulation.latencia import ParamsLatencia
from src.simulation.requisicoes import ResultadoRequisicao, rodar_cenario

CAMINHO_PADRAO = Path(__file__).resolve().parents[2] / "config" / "grade.toml"

_CHAVES = {
    "geral": {"seed_base", "capacidade", "n_requisicoes", "n_aquecimento", "n_replicacoes"},
    "latencia": {"mediana_ms", "sigma", "p_corte"},
    "grade": {"fan_out", "concorrencia", "xi"},
}


@dataclass(frozen=True)
class Cenario:
    """Um ponto da grade: fan-out, concorrência e peso da cauda, mais o que é comum."""

    n_fan_out: int
    concorrencia: int
    xi: float
    capacidade: int
    n_requisicoes: int
    n_aquecimento: int
    n_replicacoes: int
    seed_base: int
    mediana_ms: float
    sigma: float
    p_corte: float

    @property
    def id(self) -> str:
        return f"N{self.n_fan_out:03d}_C{self.concorrencia:02d}_xi{self.xi:.2f}"

    @property
    def params(self) -> ParamsLatencia:
        return ParamsLatencia(
            mu=math.log(self.mediana_ms),
            sigma=self.sigma,
            p_corte=self.p_corte,
            xi=self.xi,
        )

    @property
    def sem_contencao(self) -> bool:
        """Sem fila: no máximo `capacidade` requisições simultâneas por API."""
        return self.concorrencia <= self.capacidade

    def seed(self, replicacao: int) -> int:
        """Seed da replicação, derivada do cenário (estável e sem colisões)."""
        if not 0 <= replicacao < self.n_replicacoes:
            raise ValueError(
                f"replicacao deve estar entre 0 e {self.n_replicacoes - 1}, recebeu {replicacao}"
            )
        entropia = [
            self.seed_base,
            self.capacidade,
            self.n_fan_out,
            self.concorrencia,
            round(100 * self.xi),
            replicacao,
        ]
        return int(np.random.SeedSequence(entropia).generate_state(1)[0])


def _inteiro(valor, nome: str, minimo: int) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < minimo:
        raise ValueError(f"{nome} deve ser um inteiro >= {minimo}, recebeu {valor!r}")
    return valor


def _real(valor, nome: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or valor <= 0:
        raise ValueError(f"{nome} deve ser um número > 0, recebeu {valor!r}")
    return float(valor)


def _lista(valores, nome: str) -> list:
    if not isinstance(valores, list) or not valores:
        raise ValueError(f"{nome} deve ser uma lista não vazia, recebeu {valores!r}")
    if len(set(valores)) != len(valores):
        raise ValueError(f"{nome} tem valores repetidos: {valores!r}")
    return valores


def carregar_grade(caminho: Path | str = CAMINHO_PADRAO) -> tuple[Cenario, ...]:
    """Lê e valida o TOML e devolve os cenários, na ordem (N, C, xi).

    Falha com ValueError se faltar uma seção ou chave, se houver chave
    desconhecida (erro de digitação) ou se algum valor for inválido.
    """
    with open(caminho, "rb") as f:
        dados = tomllib.load(f)

    if set(dados) != set(_CHAVES):
        raise ValueError(f"seções esperadas {sorted(_CHAVES)}, encontradas {sorted(dados)}")
    for secao, esperadas in _CHAVES.items():
        encontradas = set(dados[secao])
        if encontradas != esperadas:
            raise ValueError(
                f"[{secao}]: faltam {sorted(esperadas - encontradas)}, "
                f"sobram {sorted(encontradas - esperadas)}"
            )

    g, lat, grade = dados["geral"], dados["latencia"], dados["grade"]
    comum = dict(
        seed_base=_inteiro(g["seed_base"], "seed_base", 0),
        capacidade=_inteiro(g["capacidade"], "capacidade", 1),
        n_requisicoes=_inteiro(g["n_requisicoes"], "n_requisicoes", 1),
        n_aquecimento=_inteiro(g["n_aquecimento"], "n_aquecimento", 0),
        n_replicacoes=_inteiro(g["n_replicacoes"], "n_replicacoes", 1),
        mediana_ms=_real(lat["mediana_ms"], "mediana_ms"),
        sigma=_real(lat["sigma"], "sigma"),
        p_corte=_real(lat["p_corte"], "p_corte"),
    )
    if not 0 < comum["p_corte"] < 1:
        raise ValueError(f"p_corte deve estar entre 0 e 1, recebeu {comum['p_corte']}")

    fan_outs = [_inteiro(v, "fan_out", 1) for v in _lista(grade["fan_out"], "fan_out")]
    concs = [_inteiro(v, "concorrencia", 1) for v in _lista(grade["concorrencia"], "concorrencia")]
    xis = [_real(v, "xi") for v in _lista(grade["xi"], "xi")]
    for xi in xis:
        if abs(100 * xi - round(100 * xi)) > 1e-9:
            raise ValueError(f"xi deve ter no máximo 2 casas decimais, recebeu {xi}")

    return tuple(
        Cenario(n_fan_out=n, concorrencia=c, xi=xi, **comum)
        for n, c, xi in product(fan_outs, concs, xis)
    )


def buscar_cenario(cenarios, id_cenario: str) -> Cenario:
    for c in cenarios:
        if c.id == id_cenario:
            return c
    raise KeyError(f"cenário {id_cenario!r} não está na grade")


def rodar_replicacao(cenario: Cenario, replicacao: int) -> list[ResultadoRequisicao]:
    """Roda uma replicação do cenário (sistema fechado, requisições medidas)."""
    return rodar_cenario(
        n_fan_out=cenario.n_fan_out,
        concorrencia=cenario.concorrencia,
        n_requisicoes=cenario.n_requisicoes,
        params=cenario.params,
        capacidade=cenario.capacidade,
        seed=cenario.seed(replicacao),
        n_aquecimento=cenario.n_aquecimento,
    )


def _resumo(cenario: Cenario, replicacao: int) -> str:
    resultado = rodar_replicacao(cenario, replicacao)
    limiar = cenario.params.quantil(0.95)  # limiar de SLA: p95 individual (seção 2.2)
    lat = np.array([r.latencia_conjunto for r in resultado])
    viol = float(np.mean(lat > limiar))
    teoria = 1 - 0.95**cenario.n_fan_out
    amp = float(np.percentile(lat, 99) / np.percentile(lat, 50))
    contencao = "sem contenção" if cenario.sem_contencao else "com contenção"
    return (
        f"{cenario.id} replicação {replicacao} (seed {cenario.seed(replicacao)}, {contencao}): "
        f"{len(resultado)} requisições; limiar p95 = {limiar:.2f} ms; "
        f"violação = {viol:.4f} (teoria sem contenção {teoria:.4f}); p99/p50 = {amp:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade de cenários do TCC")
    ap.add_argument("--grade", default=str(CAMINHO_PADRAO), help="caminho do TOML")
    acao = ap.add_mutually_exclusive_group(required=True)
    acao.add_argument("--listar", action="store_true", help="lista os cenários")
    acao.add_argument("--rodar", metavar="ID", help="roda um cenário e resume")
    ap.add_argument("--replicacao", type=int, default=0, help="com --rodar (padrão 0)")
    args = ap.parse_args()

    cenarios = carregar_grade(args.grade)
    if args.listar:
        c0 = cenarios[0]
        sem = sum(c.sem_contencao for c in cenarios)
        print(
            f"{len(cenarios)} cenários ({sem} sem contenção, {len(cenarios) - sem} com), "
            f"capacidade {c0.capacidade}, {c0.n_replicacoes} replicações de "
            f"{c0.n_requisicoes} requisições ({c0.n_aquecimento} de aquecimento): "
            f"{len(cenarios) * c0.n_replicacoes} execuções"
        )
        for c in cenarios:
            print(c.id)
    else:
        print(_resumo(buscar_cenario(cenarios, args.rodar), args.replicacao))


if __name__ == "__main__":
    main()
