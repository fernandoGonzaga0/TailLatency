"""Carga rastreável dos dados para a análise — Etapa 8 (Fase 6, item 1).

Lê o índice das execuções (`data/indice_execucoes.csv`), o dado bruto de cada
execução (`data/raw/requisicoes_<id>_rep<k>.csv.gz`) e os limiares em ms
(`data/processed/limiares.json`).

Regra de rastreabilidade: ao carregar o bruto, o SHA-256 do conteúdo
descomprimido é conferido contra o do índice; se divergir, a carga falha. Assim
a análise nunca roda sobre arquivo alterado. A análise lê o **bruto** (sem
limiar embutido) e aplica os limiares de `limiares.json` por conta própria; a
comparação com as colunas `viola_*` da Fase 5 é da conferência cruzada (item 2).

Este módulo só usa a biblioteca padrão, o pandas e `src.simulation.saida`
(também só biblioteca padrão e pandas); não importa a simulação.
"""

import csv
import json
from pathlib import Path

from src.simulation.saida import ler_requisicoes, nome_dados, sha256_conteudo

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RAW = RAIZ / "data" / "raw"
PASTA_PROCESSED = RAIZ / "data" / "processed"
CAMINHO_INDICE = RAIZ / "data" / "indice_execucoes.csv"
CAMINHO_LIMIARES = PASTA_PROCESSED / "limiares.json"

OBRIGATORIAS = (
    "id", "n_fan_out", "concorrencia", "xi", "replicacao", "seed",
    "n_requisicoes", "sha256_bruto",
)
_INTEIROS = ("n_fan_out", "concorrencia", "replicacao", "seed", "n_requisicoes")
_BOOLEANOS = ("alteracoes_locais",)
_TEXTOS = (
    "id", "sha256_bruto", "sha256_marcado", "commit", "python", "numpy", "scipy", "simpy",
)
NOMES_LIMIAR = ("p90", "p95", "p99")


def ler_indice(caminho: Path | str = CAMINHO_INDICE) -> list[dict]:
    """Lê o índice como lista de dicionários com os tipos convertidos.

    Inteiros: fan-out, concorrência, replicação, seed e número de requisições;
    textos: id, hashes, commit e versões; booleano: `alteracoes_locais`; as demais
    colunas são números (float). Falha se faltar coluna obrigatória.
    """
    with open(caminho, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        colunas = leitor.fieldnames or []
        faltam = [c for c in OBRIGATORIAS if c not in colunas]
        if faltam:
            raise ValueError(f"o índice não tem as colunas: {faltam}")
        linhas = []
        for bruta in leitor:
            linha = {}
            for chave, valor in bruta.items():
                if chave in _INTEIROS:
                    linha[chave] = int(valor)
                elif chave in _BOOLEANOS:
                    linha[chave] = valor.strip().lower() == "true"
                elif chave in _TEXTOS:
                    linha[chave] = valor
                else:
                    linha[chave] = float(valor)
            linhas.append(linha)
    if not linhas:
        raise ValueError("o índice está vazio")
    chaves = [(x["id"], x["replicacao"]) for x in linhas]
    if len(set(chaves)) != len(chaves):
        raise ValueError("o índice tem execuções repetidas")
    return linhas


def carregar_limiares(caminho: Path | str = CAMINHO_LIMIARES) -> dict:
    """Lê `limiares.json` (só leitura; não recalcula nada)."""
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    for chave in ("regra", "percentis", "limiares"):
        if chave not in dados:
            raise ValueError(f"limiares.json sem a chave {chave!r}")
    if set(dados["percentis"]) != set(NOMES_LIMIAR):
        raise ValueError(f"percentis inesperados: {sorted(dados['percentis'])}")
    return dados


def limiares_do_xi(limiares: dict, xi: float) -> dict:
    """{'p90': ms, 'p95': ms, 'p99': ms} do ξ dado (busca exata, como na marcação)."""
    for registro in limiares["limiares"]:
        if registro["xi"] == xi:
            return {n: float(registro[f"limiar_{n}_ms"]) for n in NOMES_LIMIAR}
    raise ValueError(f"xi = {xi!r} não consta no arquivo de limiares")


def caminho_bruto(linha: dict, pasta_raw: Path | str = PASTA_RAW) -> Path:
    return Path(pasta_raw) / nome_dados(linha["id"], linha["replicacao"])


def carregar_execucao(linha: dict, pasta_raw: Path | str = PASTA_RAW, verificar_hash: bool = True):
    """Lê o bruto de uma execução (DataFrame), conferindo hash e número de linhas.

    `linha` é uma linha de `ler_indice`. Falha com FileNotFoundError se o arquivo
    não existe e com ValueError se o SHA-256 ou o número de linhas diverge do
    índice. `verificar_hash=False` só para testes.
    """
    caminho = caminho_bruto(linha, pasta_raw)
    if not caminho.exists():
        raise FileNotFoundError(f"dado bruto ausente: {caminho}")
    rotulo = f"{linha['id']} rep{linha['replicacao']}"
    if verificar_hash:
        achado = sha256_conteudo(caminho)
        if achado != linha["sha256_bruto"]:
            raise ValueError(f"{rotulo}: SHA-256 do bruto diverge do índice")
    tabela = ler_requisicoes(caminho)
    if len(tabela) != linha["n_requisicoes"]:
        raise ValueError(
            f"{rotulo}: {len(tabela)} linhas no arquivo, {linha['n_requisicoes']} no índice"
        )
    return tabela
