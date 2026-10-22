"""Formato de saída dos dados brutos — Etapa 5, passo 7.

Decisões (22/10/2026):

- **Nível de detalhe: uma linha por requisição**, não por chamada. Uma linha por
  chamada daria cerca de 103 milhões de linhas na grade inteira (o N = 50 sozinho
  gera 500 mil por execução); por requisição são cerca de 6 milhões. Tudo o que
  a análise precisa mora aí: percentis do conjunto, p99/p50 e as colunas de
  violação por limiar (calculadas depois, em `src/data/`, a partir de
  `latencia_conjunto_ms`; o arquivo bruto não tem limiar embutido, seção 2.2).
- **Formato: CSV comprimido (`.csv.gz`), um arquivo por execução**, ao lado do
  manifesto. Sem dependência nova (Parquet exigiria `pyarrow`), legível, lido
  direto por `pandas.read_csv`, com floats gravados com precisão completa
  (`repr`), então a releitura devolve exatamente os mesmos valores. Um arquivo
  por execução permite rodar em paralelo e retomar pulando o que já existe.
- A gravação é determinística: quebras de linha `\\n` em qualquer sistema e
  cabeçalho gzip sem nome nem data, então a mesma execução gera o mesmo arquivo,
  byte a byte. O manifesto guarda o SHA-256 do conteúdo descomprimido.
- O arquivo é gravado em um temporário e renomeado ao final, para uma execução
  interrompida não deixar um arquivo incompleto com o nome definitivo.

Colunas (tempo em ms, 1,0 de tempo simulado = 1 ms):

    id_requisicao, laco, inicio_ms, fim_ms, latencia_conjunto_ms,
    servico_conjunto_ms, espera_max_ms, espera_media_ms

`servico_conjunto_ms` é o máximo dos serviços sem fila: separa a violação
causada pela cauda do serviço da causada pela fila. `espera_max_ms` e
`espera_media_ms` são, entre as N chamadas da requisição, a maior e a média das
esperas na fila.

O nível de chamada (uma linha por chamada) não é gravado; se a análise precisar
dele para cenários escolhidos, decide-se na Fase 5.

Como verificar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_saida
"""

import gzip
import hashlib
import io
import os
from pathlib import Path

COLUNAS = (
    "id_requisicao",
    "laco",
    "inicio_ms",
    "fim_ms",
    "latencia_conjunto_ms",
    "servico_conjunto_ms",
    "espera_max_ms",
    "espera_media_ms",
)


def nome_dados(id_cenario: str, replicacao: int) -> str:
    return f"requisicoes_{id_cenario}_rep{replicacao}.csv.gz"


def _linha(r) -> str:
    esperas = [c.espera for c in r.chamadas]
    campos = (
        r.id_requisicao,
        r.laco,
        float(r.inicio),
        float(r.fim),
        float(r.latencia_conjunto),
        float(r.servico_conjunto),
        float(max(esperas)),
        float(sum(esperas) / len(esperas)),
    )
    return ",".join(repr(c) if isinstance(c, float) else str(c) for c in campos)


def conteudo_csv(resultado) -> bytes:
    """CSV em UTF-8, com cabeçalho e quebras de linha '\\n'."""
    linhas = [",".join(COLUNAS)] + [_linha(r) for r in resultado]
    return ("\n".join(linhas) + "\n").encode("utf-8")


def gravar_requisicoes(resultado, caminho: Path | str) -> dict:
    """Grava a tabela por requisição e devolve {arquivo, linhas, sha256_conteudo}."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conteudo = conteudo_csv(resultado)
    temporario = caminho.with_name(caminho.name + ".tmp")
    with open(temporario, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
            gz.write(conteudo)
    os.replace(temporario, caminho)
    return {
        "arquivo": caminho.name,
        "linhas": len(resultado),
        "sha256_conteudo": hashlib.sha256(conteudo).hexdigest(),
    }


def sha256_conteudo(caminho: Path | str) -> str:
    """SHA-256 do conteúdo descomprimido do arquivo de dados."""
    with gzip.open(caminho, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ler_requisicoes(caminho: Path | str):
    """Lê a tabela por requisição como DataFrame do pandas."""
    import pandas as pd

    with gzip.open(caminho, "rb") as f:
        # round_trip: o leitor rápido padrão pode errar o último bit dos floats
        return pd.read_csv(io.BytesIO(f.read()), float_precision="round_trip")
