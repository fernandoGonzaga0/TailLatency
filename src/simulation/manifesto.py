"""Reprodutibilidade — Etapa 5, passo 6 (com os dados brutos do passo 7).

Cada execução (cenário x replicação) gera um manifesto em JSON, gravado ao lado
dos dados brutos, com tudo o que permite reconstruí-la e conferi-la depois:

- o cenário completo, a seed e os valores derivados (u, beta e os limiares de
  SLA em ms: p90, p95 e p99 da chamada individual, seção 2.2);
- o SHA-256 do `config/grade.toml` (com quebras de linha normalizadas, para o
  mesmo arquivo dar o mesmo hash no Windows e no Linux);
- as versões de Python, NumPy, SciPy e SimPy, o commit do git e se havia
  alterações locais, a data (UTC) e a duração;
- o arquivo de dados brutos da execução (uma linha por requisição, ver
  `saida.py`): nome, número de linhas e SHA-256 do conteúdo descomprimido;
- um resumo do resultado (violação do p95, p50, p99, p99/p50, latência e espera
  médias).

Verificação
-----------
`verificar_manifesto` reconstrói o cenário a partir do próprio manifesto (não da
grade atual), roda a replicação de novo e compara o resumo. Não se compara hash
dos dados refeitos: o SciPy calcula o quantil da normal em ponto flutuante e
versões diferentes podem divergir na casa de 1e-15, o que mudaria o hash sem
nada errado. A comparação tem tolerância: a fração de violação pode diferir em
no máximo 1 requisição (1/n) e as demais medidas, em 1e-6 relativo. Mudança de
versão ou de arquivo de grade é só aviso; divergência no resumo ou na seed é
falha.

Com `pasta_dados`, confere também o arquivo de dados gravado: ele existe, tem o
número de linhas e o SHA-256 do manifesto (isso é exato, pois é o mesmo arquivo)
e o resumo recalculado a partir dele bate com o do manifesto.

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.manifesto --gerar N010_C06_xi0.50 --replicacao 0
    python -m src.simulation.manifesto --verificar data/raw/manifesto_N010_C06_xi0.50_rep0.json
    python -m src.simulation.teste_manifesto

Manifestos e dados ficam em `data/raw/` por padrão, que o `.gitignore` ignora.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from src.simulation.cenarios import (
    CAMINHO_PADRAO,
    Cenario,
    buscar_cenario,
    carregar_grade,
    rodar_replicacao,
)
from src.simulation.saida import (
    gravar_requisicoes,
    ler_requisicoes,
    nome_dados,
    sha256_conteudo,
)

RAIZ = Path(__file__).resolve().parents[2]
PASTA_PADRAO = RAIZ / "data" / "raw"
VERSAO_MANIFESTO = 2  # 2: acrescenta o campo "dados" (passo 7)

# Tolerâncias da verificação (ver a docstring do módulo).
TOL_RELATIVA = 1e-6
TOL_ABSOLUTA = 1e-9


def sha256_grade(caminho: Path | str) -> str:
    """SHA-256 do arquivo, normalizando CRLF para LF."""
    dados = Path(caminho).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(dados).hexdigest()


def _versao(pacote: str) -> str | None:
    try:
        return metadata.version(pacote)
    except metadata.PackageNotFoundError:
        return None


def _info_ambiente() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": _versao("numpy"),
        "scipy": _versao("scipy"),
        "simpy": _versao("simpy"),
        "plataforma": platform.platform(),
    }


def _info_git(raiz: Path = RAIZ) -> dict:
    """Commit atual e se há alterações locais; None se não houver git ou repositório."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True, timeout=10
        )
        if head.returncode != 0:
            return {"commit": None, "alteracoes_locais": None}
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=raiz, capture_output=True, text=True, timeout=10
        )
        return {
            "commit": head.stdout.strip(),
            "alteracoes_locais": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "alteracoes_locais": None}


def resumir_arrays(cenario: Cenario, latencia, espera_media) -> dict:
    """Resumo a partir das latências do conjunto e da espera média por requisição.

    Como cada requisição tem N chamadas, a média das esperas médias por
    requisição é a espera média por chamada.
    """
    limiar = cenario.params.quantil(0.95)
    lat = np.asarray(latencia, dtype=float)
    p50 = float(np.percentile(lat, 50))
    p99 = float(np.percentile(lat, 99))
    return {
        "n_requisicoes": int(len(lat)),
        "violacao_p95": float(np.mean(lat > limiar)),
        "p50_ms": p50,
        "p99_ms": p99,
        "amplificacao_p99_p50": p99 / p50,
        "latencia_media_ms": float(np.mean(lat)),
        "espera_media_chamada_ms": float(np.mean(np.asarray(espera_media, dtype=float))),
    }


def resumir(cenario: Cenario, resultado) -> dict:
    """Resumo do resultado de uma replicação (usado na verificação)."""
    lat = [r.latencia_conjunto for r in resultado]
    espera_media = [sum(c.espera for c in r.chamadas) / len(r.chamadas) for r in resultado]
    return resumir_arrays(cenario, lat, espera_media)


def resumir_dados(cenario: Cenario, tabela) -> dict:
    """Resumo a partir da tabela por requisição lida de um arquivo de dados."""
    return resumir_arrays(
        cenario, tabela["latencia_conjunto_ms"].to_numpy(), tabela["espera_media_ms"].to_numpy()
    )


def gerar_manifesto(
    cenario: Cenario,
    replicacao: int,
    resultado,
    duracao_s: float,
    caminho_grade: Path | str = CAMINHO_PADRAO,
    inicio_utc: str | None = None,
    dados: dict | None = None,
) -> dict:
    p = cenario.params
    return {
        "versao_manifesto": VERSAO_MANIFESTO,
        "id": cenario.id,
        "replicacao": replicacao,
        "seed": cenario.seed(replicacao),
        "cenario": dataclasses.asdict(cenario),
        "derivados": {
            "u_ms": p.u,
            "beta_ms": p.beta,
            "limiar_p90_ms": p.quantil(0.90),
            "limiar_p95_ms": p.quantil(0.95),
            "limiar_p99_ms": p.quantil(0.99),
        },
        "grade": {"arquivo": Path(caminho_grade).name, "sha256": sha256_grade(caminho_grade)},
        "ambiente": _info_ambiente(),
        "git": _info_git(),
        "execucao": {
            "inicio_utc": inicio_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duracao_s": round(duracao_s, 3),
        },
        "dados": dados,
        "resumo": resumir(cenario, resultado),
    }


def nome_manifesto(cenario: Cenario, replicacao: int) -> str:
    return f"manifesto_{cenario.id}_rep{replicacao}.json"


def gravar_manifesto(manifesto: dict, pasta: Path | str = PASTA_PADRAO) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"manifesto_{manifesto['id']}_rep{manifesto['replicacao']}.json"
    caminho.write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return caminho


def carregar_manifesto(caminho: Path | str) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def executar_e_registrar(
    cenario: Cenario,
    replicacao: int,
    pasta: Path | str = PASTA_PADRAO,
    caminho_grade: Path | str = CAMINHO_PADRAO,
    gravar_dados: bool = True,
):
    """Roda a replicação, grava os dados e o manifesto; devolve (resultado, caminho do manifesto).

    Os dados vão para `requisicoes_<id>_rep<k>.csv.gz` na mesma pasta (passo 7).
    A duração registrada é só a da simulação, sem a gravação.
    """
    inicio = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    resultado = rodar_replicacao(cenario, replicacao)
    duracao = time.perf_counter() - t0
    dados = None
    if gravar_dados:
        dados = gravar_requisicoes(resultado, Path(pasta) / nome_dados(cenario.id, replicacao))
    manifesto = gerar_manifesto(
        cenario, replicacao, resultado, duracao, caminho_grade, inicio, dados
    )
    return resultado, gravar_manifesto(manifesto, pasta)


def _comparar_resumos(antigo: dict, novo: dict, rotulo_novo: str) -> list[str]:
    diferencas = []
    n = antigo["n_requisicoes"]
    for chave, valor in antigo.items():
        atual = novo[chave]
        if chave == "n_requisicoes":
            ok = atual == valor
        elif chave == "violacao_p95":
            ok = abs(atual - valor) <= 1.0 / n + 1e-12  # no máximo 1 requisição
        else:
            ok = math.isclose(atual, valor, rel_tol=TOL_RELATIVA, abs_tol=TOL_ABSOLUTA)
        if not ok:
            diferencas.append(f"{chave}: gravado {valor!r}, {rotulo_novo} {atual!r}")
    return diferencas


def verificar_dados(manifesto: dict, pasta: Path | str) -> list[str]:
    """Confere o arquivo de dados citado no manifesto; devolve a lista de problemas."""
    dados = manifesto.get("dados")
    if dados is None:
        return []
    caminho = Path(pasta) / dados["arquivo"]
    if not caminho.exists():
        return [f"arquivo de dados ausente: {caminho.name}"]
    problemas = []
    if sha256_conteudo(caminho) != dados["sha256_conteudo"]:
        problemas.append(f"{caminho.name}: o SHA-256 do conteúdo difere do manifesto")
    tabela = ler_requisicoes(caminho)
    if len(tabela) != dados["linhas"]:
        problemas.append(f"{caminho.name}: {len(tabela)} linhas, o manifesto diz {dados['linhas']}")
    else:
        cenario = Cenario(**manifesto["cenario"])
        problemas += [
            f"{caminho.name}: {d}"
            for d in _comparar_resumos(
                manifesto["resumo"], resumir_dados(cenario, tabela), "no arquivo"
            )
        ]
    return problemas


def verificar_manifesto(
    manifesto: dict,
    caminho_grade: Path | str | None = None,
    pasta_dados: Path | str | None = None,
) -> dict:
    """Roda a replicação de novo a partir do manifesto e compara o resumo.

    Devolve {"ok": bool, "diferencas": [...], "avisos": [...]}. Diferenças (seed,
    resumo ou arquivo de dados) tornam ok = False; avisos (versões, grade) não.
    Com `pasta_dados`, confere também o arquivo de dados gravado.
    """
    diferencas, avisos = [], []
    cenario = Cenario(**manifesto["cenario"])
    k = manifesto["replicacao"]

    if cenario.id != manifesto["id"]:
        diferencas.append(f"id {manifesto['id']} não corresponde ao cenário {cenario.id}")
    if cenario.seed(k) != manifesto["seed"]:
        diferencas.append(
            f"seed derivada {cenario.seed(k)} difere da gravada {manifesto['seed']}"
        )

    novo = resumir(cenario, rodar_replicacao(cenario, k))
    diferencas += _comparar_resumos(manifesto["resumo"], novo, "refeito")

    amb_atual, amb_antigo = _info_ambiente(), manifesto.get("ambiente", {})
    for pacote in ("python", "numpy", "scipy", "simpy"):
        if amb_atual[pacote] != amb_antigo.get(pacote):
            avisos.append(f"{pacote}: gravado {amb_antigo.get(pacote)}, agora {amb_atual[pacote]}")
    if caminho_grade is not None:
        if sha256_grade(caminho_grade) != manifesto["grade"]["sha256"]:
            avisos.append("o arquivo de grade mudou desde a execução (hash diferente)")
    if pasta_dados is not None:
        diferencas += verificar_dados(manifesto, pasta_dados)

    return {"ok": not diferencas, "diferencas": diferencas, "avisos": avisos}


def main() -> None:
    ap = argparse.ArgumentParser(description="Manifestos de reprodutibilidade")
    acao = ap.add_mutually_exclusive_group(required=True)
    acao.add_argument("--gerar", metavar="ID", help="roda o cenário e grava dados e manifesto")
    acao.add_argument("--verificar", metavar="CAMINHO", help="refaz a execução e compara")
    ap.add_argument("--replicacao", type=int, default=0, help="com --gerar (padrão 0)")
    ap.add_argument("--pasta", default=str(PASTA_PADRAO), help="com --gerar")
    ap.add_argument("--sem-dados", action="store_true", help="com --gerar: só o manifesto")
    ap.add_argument("--grade", default=str(CAMINHO_PADRAO), help="caminho do TOML")
    args = ap.parse_args()

    if args.gerar:
        cenario = buscar_cenario(carregar_grade(args.grade), args.gerar)
        _, caminho = executar_e_registrar(
            cenario, args.replicacao, args.pasta, args.grade, gravar_dados=not args.sem_dados
        )
        print(f"manifesto gravado em {caminho}")
    else:
        caminho = Path(args.verificar)
        v = verificar_manifesto(carregar_manifesto(caminho), args.grade, caminho.parent)
        for a in v["avisos"]:
            print(f"aviso: {a}")
        for d in v["diferencas"]:
            print(f"DIFERENÇA: {d}")
        print("reprodução confere" if v["ok"] else "reprodução NÃO confere")
        raise SystemExit(0 if v["ok"] else 1)


if __name__ == "__main__":
    main()
