"""Marcação por limiar de SLA — Etapa 6 (Fase 5 do checklist).

Lê o dado bruto de cada execução (`data/raw/requisicoes_<id>_rep<k>.csv.gz`, sem
limiar embutido) e grava em `data/processed/` a mesma tabela com três colunas a
mais, uma por limiar (seção 2.2 do documento de contexto):

    viola_p90, viola_p95, viola_p99     (1 = a requisição violou, 0 = não)

Cada coluna compara a latência do conjunto (a mais lenta das N chamadas) com o
limiar de referência fixa, lido de `data/processed/limiares.json` (ver
`limiares.py`): `viola_pXX = latencia_conjunto_ms > limiar_pXX_ms`.

Como a marcação acontece em duas fases do projeto
-------------------------------------------------
- **Fase 5 (aqui):** grava o dado marcado em `data/processed/` e, ao lado, um
  registro `marcacao_<id>_rep<k>.json` com o SHA-256 do bruto e do marcado, os
  limiares usados e a fração de violação por limiar. Dá rastreabilidade por
  arquivo.
- **Fase 6 (`src/analysis/`):** a análise de sensibilidade aplica os limiares de
  novo sobre `latencia_conjunto_ms` (o bruto continua sem limiar), calcula os IC
  entre replicações e confere que os dois caminhos concordam com o que foi
  gravado aqui. Qualquer limiar novo pode ser aplicado sem rodar a simulação.

Detalhes de projeto
-------------------
- As colunas originais são copiadas **como texto**, sem reler e regravar os
  floats; então ficam idênticas byte a byte às do bruto. Só se lê
  `latencia_conjunto_ms` para comparar com o limiar.
- A gravação é a mesma do dado bruto (`saida.gravar_conteudo`): gzip sem nome
  nem data, `\\n`, temporário e renomeação. A mesma marcação gera o mesmo
  arquivo, byte a byte.
- Os limiares vêm do arquivo de metadados, não são recalculados; antes de marcar
  confere-se que batem com os `derivados` do manifesto da execução (1e-9
  relativo). Se não baterem, a marcação falha.
- O arquivo bruto também é conferido: o SHA-256 do conteúdo tem de ser o do
  manifesto.

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.data.limiares --gerar
    python -m src.data.marcacao --marcar data/raw/manifesto_N010_C06_xi0.50_rep0.json
    python -m src.data.marcacao --tudo
    python -m src.data.marcacao --verificar data/processed/marcacao_N010_C06_xi0.50_rep0.json
    python -m src.data.teste_marcacao
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from src.data.limiares import (
    NOME_ARQUIVO,
    PERCENTIS,
    carregar_limiares,
    limiares_do_cenario,
    sha256_limiares,
)
from src.data.limiares import PASTA_PADRAO as PASTA_PROCESSED
from src.simulation.manifesto import carregar_manifesto
from src.simulation.saida import COLUNAS, gravar_conteudo, nome_dados, sha256_conteudo

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RAW = RAIZ / "data" / "raw"
VERSAO = 1

COLUNAS_VIOLACAO = tuple(f"viola_{nome}" for nome in PERCENTIS)  # viola_p90, viola_p95, viola_p99
COLUNA_LATENCIA = COLUNAS.index("latencia_conjunto_ms")
TOL_LIMIAR = 1e-9  # relativa, ao conferir os limiares do arquivo contra o manifesto


def nome_registro(id_cenario: str, replicacao: int) -> str:
    return f"marcacao_{id_cenario}_rep{replicacao}.json"


def marcar_conteudo(bruto: bytes, limiares_ms: dict) -> tuple[bytes, dict]:
    """Acrescenta as colunas de violação ao CSV bruto.

    `limiares_ms` é {'p90': ms, 'p95': ms, 'p99': ms}. Devolve o CSV marcado e a
    contagem de violações por limiar: {'linhas': n, 'p90': k, 'p95': k, 'p99': k}.
    Violação é latência estritamente acima do limiar.
    """
    linhas = bruto.decode("utf-8").split("\n")
    if linhas[-1] != "":
        raise ValueError("o CSV bruto deve terminar em quebra de linha")
    cabecalho, corpo = linhas[0], linhas[1:-1]
    if cabecalho != ",".join(COLUNAS):
        raise ValueError(f"cabeçalho inesperado no CSV bruto: {cabecalho!r}")
    nomes = list(PERCENTIS)
    valores = [limiares_ms[nome] for nome in nomes]
    contagem = {nome: 0 for nome in nomes}
    saida = [cabecalho + "," + ",".join(COLUNAS_VIOLACAO)]
    for linha in corpo:
        lat = float(linha.split(",")[COLUNA_LATENCIA])
        marcas = [lat > limiar for limiar in valores]
        for nome, marca in zip(nomes, marcas):
            contagem[nome] += marca
        saida.append(linha + "," + ",".join("1" if m else "0" for m in marcas))
    contagem["linhas"] = len(corpo)
    return ("\n".join(saida) + "\n").encode("utf-8"), contagem


def _ler_gz(caminho: Path) -> bytes:
    with gzip.open(caminho, "rb") as f:
        return f.read()


def _fracoes(contagem: dict) -> dict:
    n = contagem["linhas"]
    return {nome: contagem[nome] / n for nome in PERCENTIS}


def _conferir_limiares_com_manifesto(limiares_ms: dict, manifesto: dict) -> None:
    derivados = manifesto["derivados"]
    for nome in PERCENTIS:
        esperado = derivados[f"limiar_{nome}_ms"]
        if not math.isclose(limiares_ms[nome], esperado, rel_tol=TOL_LIMIAR):
            raise ValueError(
                f"{manifesto['id']}: limiar {nome} do arquivo de limiares ({limiares_ms[nome]!r}) "
                f"difere do manifesto ({esperado!r})"
            )


def marcar_execucao(
    caminho_manifesto: Path | str,
    pasta_saida: Path | str = PASTA_PROCESSED,
    caminho_limiares: Path | str | None = None,
) -> Path:
    """Marca uma execução e devolve o caminho do registro `marcacao_<id>_rep<k>.json`.

    O dado bruto é lido da pasta do manifesto. Falha com ValueError se o bruto não
    existir ou não bater com o SHA-256 do manifesto, se o xi não constar nos
    limiares ou se os limiares divergirem dos do manifesto.
    """
    caminho_manifesto = Path(caminho_manifesto)
    pasta_saida = Path(pasta_saida)
    caminho_limiares = Path(caminho_limiares or pasta_saida / NOME_ARQUIVO)
    manifesto = carregar_manifesto(caminho_manifesto)
    dados = manifesto.get("dados")
    if dados is None:
        raise ValueError(f"{manifesto['id']}: o manifesto não tem arquivo de dados")

    caminho_bruto = caminho_manifesto.parent / dados["arquivo"]
    if not caminho_bruto.exists():
        raise ValueError(f"arquivo bruto ausente: {caminho_bruto.name}")
    bruto = _ler_gz(caminho_bruto)
    sha_bruto = hashlib.sha256(bruto).hexdigest()
    if sha_bruto != dados["sha256_conteudo"]:
        raise ValueError(f"{caminho_bruto.name}: o SHA-256 do conteúdo difere do manifesto")

    limiares = carregar_limiares(caminho_limiares)
    limiares_ms = limiares_do_cenario(limiares, manifesto["cenario"])
    _conferir_limiares_com_manifesto(limiares_ms, manifesto)

    marcado, contagem = marcar_conteudo(bruto, limiares_ms)
    if contagem["linhas"] != dados["linhas"]:
        raise ValueError(f"{caminho_bruto.name}: número de linhas difere do manifesto")
    fracoes = _fracoes(contagem)
    n = contagem["linhas"]
    ref = manifesto["resumo"]["violacao_p95"]
    if abs(fracoes["p95"] - ref) > 1.0 / n + 1e-12:
        raise ValueError(
            f"{manifesto['id']}: violação p95 marcada ({fracoes['p95']}) difere do resumo "
            f"do manifesto ({ref})"
        )

    nome = nome_dados(manifesto["id"], manifesto["replicacao"])
    gravar_conteudo(marcado, pasta_saida / nome)
    registro = {
        "versao_marcacao": VERSAO,
        "id": manifesto["id"],
        "replicacao": manifesto["replicacao"],
        "regra": "viola_pXX = latencia_conjunto_ms > limiar_pXX_ms",
        "bruto": {"arquivo": dados["arquivo"], "sha256_conteudo": sha_bruto},
        "marcado": {
            "arquivo": nome,
            "linhas": n,
            "colunas": list(COLUNAS + COLUNAS_VIOLACAO),
            "sha256_conteudo": hashlib.sha256(marcado).hexdigest(),
        },
        "limiares_ms": limiares_ms,
        "limiares_arquivo": {"nome": caminho_limiares.name, "sha256": sha256_limiares(caminho_limiares)},
        "violacoes": {nome_p: contagem[nome_p] for nome_p in PERCENTIS},
        "fracao_violacao": fracoes,
    }
    caminho = pasta_saida / nome_registro(manifesto["id"], manifesto["replicacao"])
    caminho.write_text(
        json.dumps(registro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return caminho


def carregar_registro(caminho: Path | str) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def verificar_marcacao(
    caminho_registro: Path | str,
    pasta_bruta: Path | str = PASTA_RAW,
    caminho_limiares: Path | str | None = None,
) -> dict:
    """Confere um registro de marcação; devolve {'ok', 'diferencas', 'avisos'}.

    Refaz a marcação a partir do bruto, usando os limiares **gravados no registro**,
    e compara byte a byte (SHA-256) com o arquivo marcado. Como é só comparação de
    texto, a conferência é exata. Se `caminho_limiares` for dado e o arquivo de
    limiares mudou desde a marcação, é só aviso.
    """
    caminho_registro = Path(caminho_registro)
    pasta_saida = caminho_registro.parent
    pasta_bruta = Path(pasta_bruta)
    r = carregar_registro(caminho_registro)
    diferencas, avisos = [], []

    caminho_bruto = pasta_bruta / r["bruto"]["arquivo"]
    caminho_marcado = pasta_saida / r["marcado"]["arquivo"]
    if not caminho_bruto.exists():
        return {"ok": False, "diferencas": [f"arquivo bruto ausente: {caminho_bruto.name}"], "avisos": avisos}
    if not caminho_marcado.exists():
        return {"ok": False, "diferencas": [f"arquivo marcado ausente: {caminho_marcado.name}"], "avisos": avisos}

    if sha256_conteudo(caminho_bruto) != r["bruto"]["sha256_conteudo"]:
        diferencas.append(f"{caminho_bruto.name}: o SHA-256 do bruto difere do registro")
    if sha256_conteudo(caminho_marcado) != r["marcado"]["sha256_conteudo"]:
        diferencas.append(f"{caminho_marcado.name}: o SHA-256 do marcado difere do registro")

    if not diferencas:
        refeito, contagem = marcar_conteudo(_ler_gz(caminho_bruto), r["limiares_ms"])
        if hashlib.sha256(refeito).hexdigest() != r["marcado"]["sha256_conteudo"]:
            diferencas.append("a marcação refeita a partir do bruto difere do arquivo marcado")
        for nome in PERCENTIS:
            if contagem[nome] != r["violacoes"][nome]:
                diferencas.append(
                    f"violações em {nome}: registro {r['violacoes'][nome]}, refeito {contagem[nome]}"
                )

    if caminho_limiares is not None and Path(caminho_limiares).exists():
        if sha256_limiares(caminho_limiares) != r["limiares_arquivo"]["sha256"]:
            avisos.append("o arquivo de limiares mudou desde a marcação (hash diferente)")
    return {"ok": not diferencas, "diferencas": diferencas, "avisos": avisos}


def marcar_tudo(
    pasta_bruta: Path | str = PASTA_RAW,
    pasta_saida: Path | str = PASTA_PROCESSED,
    caminho_limiares: Path | str | None = None,
    progresso=print,
) -> dict:
    """Marca todas as execuções com manifesto em `pasta_bruta`; retomável.

    Uma execução cujo registro já existe e confere é pulada. Devolve
    {'marcadas': n, 'puladas': n, 'falhas': [(manifesto, motivo), ...]}.
    """
    pasta_bruta, pasta_saida = Path(pasta_bruta), Path(pasta_saida)
    caminho_limiares = Path(caminho_limiares or pasta_saida / NOME_ARQUIVO)
    manifestos = sorted(pasta_bruta.glob("manifesto_*.json"))
    marcadas, puladas, falhas = 0, 0, []
    for i, caminho in enumerate(manifestos, start=1):
        m = carregar_manifesto(caminho)
        registro = pasta_saida / nome_registro(m["id"], m["replicacao"])
        try:
            if registro.exists() and verificar_marcacao(registro, pasta_bruta, caminho_limiares)["ok"]:
                puladas += 1
            else:
                marcar_execucao(caminho, pasta_saida, caminho_limiares)
                marcadas += 1
        except (ValueError, OSError) as erro:
            falhas.append((caminho.name, str(erro)))
        progresso(f"[{i}/{len(manifestos)}] {caminho.name}")
    return {"marcadas": marcadas, "puladas": puladas, "falhas": falhas}


def main() -> None:
    ap = argparse.ArgumentParser(description="Marcação por limiar de SLA")
    acao = ap.add_mutually_exclusive_group(required=True)
    acao.add_argument("--marcar", metavar="MANIFESTO", help="marca uma execução")
    acao.add_argument("--tudo", action="store_true", help="marca todas as execuções de data/raw")
    acao.add_argument("--verificar", metavar="REGISTRO", help="confere um registro de marcação")
    ap.add_argument("--raw", default=str(PASTA_RAW), help="pasta dos dados brutos")
    ap.add_argument("--processed", default=str(PASTA_PROCESSED), help="pasta dos dados marcados")
    args = ap.parse_args()
    limiares = Path(args.processed) / NOME_ARQUIVO

    if args.marcar:
        print(f"registro gravado em {marcar_execucao(args.marcar, args.processed, limiares)}")
    elif args.tudo:
        r = marcar_tudo(args.raw, args.processed, limiares)
        print(f"{r['marcadas']} marcadas, {r['puladas']} já conferidas, {len(r['falhas'])} falhas")
        for nome, motivo in r["falhas"]:
            print(f"FALHA {nome}: {motivo}")
        raise SystemExit(1 if r["falhas"] else 0)
    else:
        v = verificar_marcacao(args.verificar, args.raw, limiares)
        for a in v["avisos"]:
            print(f"aviso: {a}")
        for d in v["diferencas"]:
            print(f"DIFERENÇA: {d}")
        print("marcação confere" if v["ok"] else "marcação NÃO confere")
        raise SystemExit(0 if v["ok"] else 1)


if __name__ == "__main__":
    main()
