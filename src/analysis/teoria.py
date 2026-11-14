"""Comparação com a teoria de estatística de ordem — Etapa 8 (Fase 6, item 3).

Usa só `data/metricas_cenarios.csv` (média e IC t de 95% entre as replicações de
cada cenário, item 1) e os parâmetros do gerador de latência; **não lê o dado
bruto**. A teoria usa a mesma distribuição do gerador (`ParamsLatencia.quantil`).

O que se compara
----------------
Com N chamadas independentes e a latência do conjunto sendo o máximo delas:

- violação: P(conjunto > limiar_q) = 1 − q^N, com q = 0,90, 0,95 e 0,99 (o limiar
  é o quantil q da chamada individual isolada);
- quantil p do conjunto = quantil p^(1/N) da chamada individual (p = 0,50, 0,95
  e 0,99);
- fator p99/p50 do conjunto = razão dos dois quantis acima. O simulado é a média
  das razões das replicações, e a razão das médias teria um viés pequeno de
  Jensen; a comparação é aproximada, e em cauda pesada o p99 do máximo é ruidoso.

A teoria vale **sem contenção** (C <= capacidade: ninguém espera na fila, as N
latências do conjunto são independentes). Com contenção a mesma fórmula serve de
linha de base: a diferença mostra quanto a fila acrescenta (item 6 faz o veredito
da hipótese e o crescimento com C; aqui a comparação é descritiva).

Posição e critério
------------------
`posicao` diz onde o simulado está em relação à teoria: "dentro" se
|simulado − teoria| <= semilargura do IC; "acima" se o IC inteiro fica acima da
teoria; "abaixo" se fica abaixo.

**Critério de aprovação (fixado em 14/11/2026, antes de gerar a tabela):** nos 40
cenários sem contenção há 120 testes de violação (3 limiares x 40). Com IC de 95%,
cerca de 6 falham só por acaso (desvio-padrão 2,4). A validação cruzada é aprovada
se **no máximo 10 dos 120** ficam fora do IC (limite superior de 95% de uma
binomial(120; 0,05); os três limiares de um mesmo cenário usam as mesmas
requisições, então a contagem real varia um pouco mais). Passando de 10, investiga-se
o simulador; a tolerância não é ajustada. Os quantis do conjunto e o p99/p50 são só
reportados, sem aprovação ou reprovação (o ruído em cauda pesada é grande).

Saída: `data/comparacao_teoria.csv` (versionado, 120 linhas).

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.teoria --gerar
    python -m src.analysis.teste_teoria

Código de saída 0 se o critério passa; 1 se não passa.
"""

import argparse
import csv
import math
from pathlib import Path

from src.analysis.agregacao import gravar_csv
from src.analysis.carga import RAIZ
from src.analysis.metricas import NOMES_LIMIAR
from src.simulation.latencia import ParamsLatencia

CAMINHO_CENARIOS = RAIZ / "data" / "metricas_cenarios.csv"
CAMINHO_SAIDA = RAIZ / "data" / "comparacao_teoria.csv"
CAMINHO_GRADE = RAIZ / "config" / "grade.toml"

Q_VIOLACAO = {"p90": 0.90, "p95": 0.95, "p99": 0.99}
Q_QUANTIS = {"p50": 0.50, "p95": 0.95, "p99": 0.99}
LIMITE_FORA = 10
TESTES_ESPERADOS = 120  # 40 cenários sem contenção x 3 limiares (grade oficial)

METRICAS_VIOLACAO = tuple(f"violacao_{n}" for n in NOMES_LIMIAR)
METRICAS_QUANTIL = tuple(f"{q}_ms" for q in Q_QUANTIS)
METRICAS_RAZAO = ("amplificacao_p99_p50",)
METRICAS = (*METRICAS_VIOLACAO, *METRICAS_QUANTIL, *METRICAS_RAZAO)
SUFIXOS = ("_teoria", "_simulado", "_ic95", "_dif", "_posicao")
IDENTIFICACAO = ("id", "n_fan_out", "concorrencia", "xi", "sem_contencao")
COLUNAS = (*IDENTIFICACAO, *(f"{m}{s}" for m in METRICAS for s in SUFIXOS))


# --- teoria -----------------------------------------------------------------

def params_do_gerador(gerador: dict, xi: float) -> ParamsLatencia:
    """ParamsLatencia a partir de {'mediana_ms', 'sigma', 'p_corte'} e do xi do cenário."""
    return ParamsLatencia(
        mu=math.log(gerador["mediana_ms"]),
        sigma=gerador["sigma"],
        p_corte=gerador["p_corte"],
        xi=xi,
    )


def violacao_teorica(n_fan_out: int, q: float) -> float:
    """P(máximo de N chamadas > quantil q da individual) = 1 − q^N."""
    if n_fan_out < 1 or not 0 < q < 1:
        raise ValueError("n_fan_out >= 1 e 0 < q < 1")
    return 1.0 - q**n_fan_out


def quantil_conjunto(params: ParamsLatencia, n_fan_out: int, p: float) -> float:
    """Quantil p do máximo de N latências i.i.d.: quantil p^(1/N) da individual (ms)."""
    if n_fan_out < 1 or not 0 < p < 1:
        raise ValueError("n_fan_out >= 1 e 0 < p < 1")
    return float(params.quantil(p ** (1.0 / n_fan_out)))


def amplificacao_teorica(params: ParamsLatencia, n_fan_out: int) -> float:
    """p99/p50 do conjunto."""
    return quantil_conjunto(params, n_fan_out, 0.99) / quantil_conjunto(params, n_fan_out, 0.50)


def teoria_da_metrica(metrica: str, params: ParamsLatencia, n_fan_out: int) -> float:
    if metrica in METRICAS_VIOLACAO:
        return violacao_teorica(n_fan_out, Q_VIOLACAO[metrica.removeprefix("violacao_")])
    if metrica in METRICAS_QUANTIL:
        return quantil_conjunto(params, n_fan_out, Q_QUANTIS[metrica.removesuffix("_ms")])
    if metrica in METRICAS_RAZAO:
        return amplificacao_teorica(params, n_fan_out)
    raise ValueError(f"métrica desconhecida: {metrica!r}")


def posicao(teoria: float, simulado: float, semilargura: float) -> str:
    """'dentro' se a teoria está no IC do simulado; 'acima' ou 'abaixo' do contrário."""
    if abs(simulado - teoria) <= semilargura:
        return "dentro"
    return "acima" if simulado - semilargura > teoria else "abaixo"


# --- leitura, comparação, resumo ----------------------------------------------

def ler_tabela_cenarios(caminho: Path | str = CAMINHO_CENARIOS) -> list[dict]:
    """Lê `metricas_cenarios.csv` com os tipos convertidos."""
    inteiros = ("n_fan_out", "concorrencia", "n_replicacoes")
    with open(caminho, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        colunas = leitor.fieldnames or []
        precisa = [
            *IDENTIFICACAO,
            *(f"{m}{s}" for m in METRICAS for s in ("_media", "_ic95")),
        ]
        faltam = [c for c in precisa if c not in colunas]
        if faltam:
            raise ValueError(f"a tabela de cenários não tem as colunas: {faltam}")
        linhas = []
        for bruta in leitor:
            linha = {}
            for chave, valor in bruta.items():
                if chave == "id":
                    linha[chave] = valor
                elif chave in inteiros:
                    linha[chave] = int(valor)
                elif chave == "sem_contencao":
                    linha[chave] = valor.strip().lower() == "true"
                else:
                    linha[chave] = float(valor)
            linhas.append(linha)
    if not linhas:
        raise ValueError("a tabela de cenários está vazia")
    return linhas


def comparar(cenarios: list[dict], gerador: dict) -> list[dict]:
    """Uma linha por cenário com teoria, simulado, IC, diferença e posição de cada métrica."""
    saida = []
    for c in sorted(cenarios, key=lambda x: x["id"]):
        params = params_do_gerador(gerador, c["xi"])
        linha = {k: c[k] for k in IDENTIFICACAO}
        for m in METRICAS:
            teoria = teoria_da_metrica(m, params, c["n_fan_out"])
            simulado, ic = c[f"{m}_media"], c[f"{m}_ic95"]
            linha[f"{m}_teoria"] = teoria
            linha[f"{m}_simulado"] = simulado
            linha[f"{m}_ic95"] = ic
            linha[f"{m}_dif"] = simulado - teoria
            linha[f"{m}_posicao"] = posicao(teoria, simulado, ic)
        saida.append(linha)
    return saida


def _contar(linhas: list[dict], metrica: str) -> dict:
    cont = {"dentro": 0, "acima": 0, "abaixo": 0}
    for x in linhas:
        cont[x[f"{metrica}_posicao"]] += 1
    cont["total"] = len(linhas)
    return cont


def resumir(comparacao: list[dict], limite_fora: int = LIMITE_FORA) -> dict:
    """Contagens por métrica (sem e com contenção) e o resultado do critério."""
    sem = [x for x in comparacao if x["sem_contencao"]]
    com = [x for x in comparacao if not x["sem_contencao"]]
    resumo = {
        "sem_contencao": {m: _contar(sem, m) for m in METRICAS},
        "com_contencao": {m: _contar(com, m) for m in METRICAS},
    }
    testes = sum(resumo["sem_contencao"][m]["total"] for m in METRICAS_VIOLACAO)
    fora = sum(testes_m["acima"] + testes_m["abaixo"]
               for testes_m in (resumo["sem_contencao"][m] for m in METRICAS_VIOLACAO))
    resumo["criterio"] = {
        "testes": testes, "fora": fora, "limite": limite_fora, "aprovado": fora <= limite_fora,
    }
    return resumo


def carregar_gerador(grade: Path | str = CAMINHO_GRADE, limiares: Path | str | None = None) -> dict:
    """{'mediana_ms', 'sigma', 'p_corte'} de `limiares.json` (se dado) ou de `grade.toml`."""
    if limiares is not None:
        import json

        with open(limiares, encoding="utf-8") as f:
            g = json.load(f)["gerador"]
    else:
        import tomllib

        with open(grade, "rb") as f:
            g = tomllib.load(f)["latencia"]
    return {k: float(g[k]) for k in ("mediana_ms", "sigma", "p_corte")}


def gerar(caminho_cenarios: Path | str, caminho_saida: Path | str, gerador: dict) -> dict:
    """Compara, grava `comparacao_teoria.csv` e devolve o resumo."""
    comparacao = comparar(ler_tabela_cenarios(caminho_cenarios), gerador)
    gravar_csv(comparacao, COLUNAS, caminho_saida)
    return resumir(comparacao)


def _imprimir(resumo: dict) -> None:
    def linha(rotulo, c, so_dentro=False):
        if so_dentro:
            return f"  {rotulo:24s} {c['dentro']:3d}/{c['total']} dentro do IC"
        return (f"  {rotulo:24s} dentro {c['dentro']:3d} | simulado acima {c['acima']:3d}"
                f" | abaixo {c['abaixo']:3d}  (de {c['total']})")

    print("SEM CONTENÇÃO (teoria exata)")
    for m in METRICAS:
        print(linha(m, resumo["sem_contencao"][m], so_dentro=True))
    print("COM CONTENÇÃO (a teoria sem fila é a linha de base)")
    for m in METRICAS:
        print(linha(m, resumo["com_contencao"][m]))
    c = resumo["criterio"]
    print(f"CRITÉRIO: {c['fora']} de {c['testes']} testes de violação fora do IC "
          f"(limite {c['limite']}; esperado por acaso cerca de {0.05 * c['testes']:.0f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparação com a estatística de ordem (Fase 6, item 3)")
    ap.add_argument("--gerar", action="store_true", help="grava data/comparacao_teoria.csv")
    ap.add_argument("--cenarios", default=str(CAMINHO_CENARIOS))
    ap.add_argument("--saida", default=str(CAMINHO_SAIDA))
    ap.add_argument("--grade", default=str(CAMINHO_GRADE), help="TOML com a seção [latencia]")
    ap.add_argument("--limiares", default=None, help="limiares.json (alternativa à grade)")
    args = ap.parse_args()
    if not args.gerar:
        ap.error("use --gerar")

    gerador = carregar_gerador(args.grade, args.limiares)
    print(f"gerador: {gerador}")
    resumo = gerar(args.cenarios, args.saida, gerador)
    _imprimir(resumo)
    print(f"gravado: {args.saida}")
    ok = resumo["criterio"]["aprovado"]
    print("validação cruzada APROVADA" if ok else "validação cruzada NÃO aprovada: investigar o simulador")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
