"""Índice das execuções — Etapa 6 (Fase 5 do checklist).

Um arquivo CSV compacto, `data/indice_execucoes.csv`, **versionado no git**, com
uma linha por execução (cenário x replicação; 600 linhas na grade oficial). Os
dados (`data/raw/`, `data/processed/`) ficam fora do git por causa do tamanho; o
índice é o registro público do que foi gerado, e é nele que o texto do TCC cita
seeds, versões e commit.

Colunas
-------
- identificação: `id`, `n_fan_out`, `concorrencia`, `xi`, `replicacao`, `seed`;
- dados: `n_requisicoes`, `sha256_bruto` e `sha256_marcado` (SHA-256 do conteúdo
  descomprimido do bruto e do marcado);
- limiares usados, em ms: `limiar_p90_ms`, `limiar_p95_ms`, `limiar_p99_ms`;
- fração de violação por limiar, da marcação: `violacao_p90`, `violacao_p95`,
  `violacao_p99`;
- resumo do manifesto: `p50_ms`, `p99_ms`, `amplificacao_p99_p50`,
  `latencia_media_ms`, `espera_media_chamada_ms`;
- procedência: `commit`, `alteracoes_locais`, `python`, `numpy`, `scipy`, `simpy`.

O que o índice prova e o que não prova
--------------------------------------
Os hashes identificam **estes** arquivos; servem para dizer que o dado foi esse.
Eles **não** servem para provar reprodução em outra máquina, porque o SciPy pode
divergir na casa de 1e-15 entre versões e mudar o hash sem erro nenhum (decisão
do passo 6 da Fase 4). Reproduzir em outra máquina se confere pelas métricas do
resumo, com a tolerância de `manifesto.verificar_manifesto`.

Detalhes
--------
- O índice é montado a partir dos manifestos (`data/raw/`) e dos registros de
  marcação (`data/processed/`); falha se uma execução tem manifesto mas não tem
  registro de marcação, se os hashes do bruto divergem entre os dois, ou se os
  limiares do registro divergem dos do manifesto.
- Linhas em ordem de (`id`, `replicacao`); floats com `repr` (a releitura devolve
  os mesmos valores); `\\n` em qualquer sistema. O mesmo conjunto de dados gera o
  mesmo arquivo, byte a byte.
- O índice mostra, ao gerar, quantos commits e ambientes distintos aparecem; o
  esperado é um só de cada (dados homogêneos).

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.data.indice --gerar
    python -m src.data.indice --verificar
    python -m src.data.teste_indice
"""

import argparse
import csv
import io
import math
from pathlib import Path

from src.data.marcacao import PASTA_RAW, carregar_registro, nome_registro
from src.data.limiares import PASTA_PADRAO as PASTA_PROCESSED
from src.simulation.manifesto import carregar_manifesto

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_PADRAO = RAIZ / "data" / "indice_execucoes.csv"

COLUNAS = (
    "id", "n_fan_out", "concorrencia", "xi", "replicacao", "seed",
    "n_requisicoes", "sha256_bruto", "sha256_marcado",
    "limiar_p90_ms", "limiar_p95_ms", "limiar_p99_ms",
    "violacao_p90", "violacao_p95", "violacao_p99",
    "p50_ms", "p99_ms", "amplificacao_p99_p50", "latencia_media_ms", "espera_media_chamada_ms",
    "commit", "alteracoes_locais", "python", "numpy", "scipy", "simpy",
)


def _linha(manifesto: dict, registro: dict) -> dict:
    c, r, d = manifesto["cenario"], manifesto["resumo"], manifesto["derivados"]
    if registro["bruto"]["sha256_conteudo"] != manifesto["dados"]["sha256_conteudo"]:
        raise ValueError(f"{manifesto['id']}: o hash do bruto difere entre manifesto e marcação")
    for nome in ("p90", "p95", "p99"):
        if not math.isclose(registro["limiares_ms"][nome], d[f"limiar_{nome}_ms"], rel_tol=1e-9):
            raise ValueError(f"{manifesto['id']}: limiar {nome} difere entre manifesto e marcação")
    g, a = manifesto["git"], manifesto["ambiente"]
    return {
        "id": manifesto["id"],
        "n_fan_out": c["n_fan_out"],
        "concorrencia": c["concorrencia"],
        "xi": c["xi"],
        "replicacao": manifesto["replicacao"],
        "seed": manifesto["seed"],
        "n_requisicoes": r["n_requisicoes"],
        "sha256_bruto": manifesto["dados"]["sha256_conteudo"],
        "sha256_marcado": registro["marcado"]["sha256_conteudo"],
        "limiar_p90_ms": registro["limiares_ms"]["p90"],
        "limiar_p95_ms": registro["limiares_ms"]["p95"],
        "limiar_p99_ms": registro["limiares_ms"]["p99"],
        "violacao_p90": registro["fracao_violacao"]["p90"],
        "violacao_p95": registro["fracao_violacao"]["p95"],
        "violacao_p99": registro["fracao_violacao"]["p99"],
        "p50_ms": r["p50_ms"],
        "p99_ms": r["p99_ms"],
        "amplificacao_p99_p50": r["amplificacao_p99_p50"],
        "latencia_media_ms": r["latencia_media_ms"],
        "espera_media_chamada_ms": r["espera_media_chamada_ms"],
        "commit": g["commit"],
        "alteracoes_locais": g["alteracoes_locais"],
        "python": a["python"],
        "numpy": a["numpy"],
        "scipy": a["scipy"],
        "simpy": a["simpy"],
    }


def construir_indice(
    pasta_raw: Path | str = PASTA_RAW, pasta_processed: Path | str = PASTA_PROCESSED
) -> list[dict]:
    """Linhas do índice, ordenadas por (id, replicação). Falha se algo não bater."""
    pasta_raw, pasta_processed = Path(pasta_raw), Path(pasta_processed)
    manifestos = sorted(pasta_raw.glob("manifesto_*.json"))
    if not manifestos:
        raise ValueError(f"nenhum manifesto em {pasta_raw}")
    linhas = []
    for caminho in manifestos:
        m = carregar_manifesto(caminho)
        if m.get("dados") is None:
            raise ValueError(f"{caminho.name}: o manifesto não tem arquivo de dados")
        reg = pasta_processed / nome_registro(m["id"], m["replicacao"])
        if not reg.exists():
            raise ValueError(f"{caminho.name}: sem registro de marcação ({reg.name})")
        linhas.append(_linha(m, carregar_registro(reg)))
    linhas.sort(key=lambda x: (x["id"], x["replicacao"]))
    return linhas


def _texto(valor) -> str:
    if isinstance(valor, float):
        return repr(valor)
    if isinstance(valor, bool):
        return "true" if valor else "false"
    return "" if valor is None else str(valor)


def conteudo_csv(linhas: list[dict]) -> bytes:
    saida = io.StringIO(newline="")
    escritor = csv.writer(saida, lineterminator="\n")
    escritor.writerow(COLUNAS)
    for linha in linhas:
        escritor.writerow([_texto(linha[c]) for c in COLUNAS])
    return saida.getvalue().encode("utf-8")


def gravar_indice(linhas: list[dict], caminho: Path | str = CAMINHO_PADRAO) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    temporario.write_bytes(conteudo_csv(linhas))
    temporario.replace(caminho)
    return caminho


def resumo_indice(linhas: list[dict]) -> dict:
    """Quantas execuções e quantos commits e ambientes distintos aparecem."""
    ambientes = {(x["python"], x["numpy"], x["scipy"], x["simpy"]) for x in linhas}
    return {
        "execucoes": len(linhas),
        "cenarios": len({x["id"] for x in linhas}),
        "commits": sorted({str(x["commit"]) for x in linhas}),
        "com_alteracoes_locais": sum(1 for x in linhas if x["alteracoes_locais"]),
        "ambientes": sorted(ambientes),
    }


def verificar_indice(
    caminho: Path | str = CAMINHO_PADRAO,
    pasta_raw: Path | str = PASTA_RAW,
    pasta_processed: Path | str = PASTA_PROCESSED,
) -> dict:
    """Refaz o índice a partir dos dados e compara, byte a byte, com o arquivo gravado.

    Devolve {'ok': bool, 'diferencas': [...]}. Reporta linhas ausentes, a mais e
    alteradas (por execução), em vez de só dizer que o arquivo difere.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return {"ok": False, "diferencas": [f"índice ausente: {caminho.name}"]}
    try:
        refeito = construir_indice(pasta_raw, pasta_processed)
    except ValueError as erro:
        return {"ok": False, "diferencas": [f"não foi possível refazer o índice: {erro}"]}
    gravado = caminho.read_bytes()
    if gravado == conteudo_csv(refeito):
        return {"ok": True, "diferencas": []}

    diferencas = []
    leitor = csv.DictReader(io.StringIO(gravado.decode("utf-8"), newline=""))
    antigas = {(x["id"], x["replicacao"]): x for x in leitor}
    novas = {(x["id"], str(x["replicacao"])): {c: _texto(x[c]) for c in COLUNAS} for x in refeito}
    for chave in sorted(novas.keys() - antigas.keys()):
        diferencas.append(f"{chave[0]} rep{chave[1]}: execução existe nos dados e não consta no índice")
    for chave in sorted(antigas.keys() - novas.keys()):
        diferencas.append(f"{chave[0]} rep{chave[1]}: consta no índice e não existe nos dados")
    for chave in sorted(antigas.keys() & novas.keys()):
        campos = [c for c in COLUNAS if antigas[chave].get(c) != novas[chave][c]]
        if campos:
            diferencas.append(f"{chave[0]} rep{chave[1]}: difere em {', '.join(campos)}")
    if not diferencas:
        diferencas.append("o arquivo difere do refeito (formato ou ordem das linhas)")
    return {"ok": False, "diferencas": diferencas}


def main() -> None:
    ap = argparse.ArgumentParser(description="Índice das execuções")
    acao = ap.add_mutually_exclusive_group(required=True)
    acao.add_argument("--gerar", action="store_true", help="gera data/indice_execucoes.csv")
    acao.add_argument("--verificar", action="store_true", help="confere o índice contra os dados")
    ap.add_argument("--raw", default=str(PASTA_RAW), help="pasta dos dados brutos")
    ap.add_argument("--processed", default=str(PASTA_PROCESSED), help="pasta dos dados marcados")
    ap.add_argument("--indice", default=str(CAMINHO_PADRAO), help="caminho do índice")
    args = ap.parse_args()

    if args.gerar:
        linhas = construir_indice(args.raw, args.processed)
        caminho = gravar_indice(linhas, args.indice)
        r = resumo_indice(linhas)
        print(f"índice gravado em {caminho}: {r['execucoes']} execuções em {r['cenarios']} cenários")
        print(f"commits: {', '.join(c[:7] for c in r['commits'])}; "
              f"execuções com alterações locais: {r['com_alteracoes_locais']}")
        for a in r["ambientes"]:
            print("ambiente: python {}, numpy {}, scipy {}, simpy {}".format(*a))
        if len(r["commits"]) > 1 or len(r["ambientes"]) > 1 or r["com_alteracoes_locais"]:
            print("aviso: dados não homogêneos (mais de um commit ou ambiente, ou árvore suja)")
    else:
        v = verificar_indice(args.indice, args.raw, args.processed)
        for d in v["diferencas"]:
            print(f"DIFERENÇA: {d}")
        print("índice confere" if v["ok"] else "índice NÃO confere")
        raise SystemExit(0 if v["ok"] else 1)


if __name__ == "__main__":
    main()
