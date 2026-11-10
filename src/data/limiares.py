"""Metadados dos limiares de SLA — Etapa 6 (Fase 5 do checklist), passo de marcação.

Os limiares são p90, p95 e p99 da latência de **uma chamada individual isolada**
(sem fan-out e sem contenção), calculados em forma fechada a partir dos
parâmetros do gerador, sem simular (seção 2.2 do documento de contexto):

- referência: p95; sensibilidade: p90 e p99;
- referência fixa: o valor em ms vem do cenário base e não é recalculado por
  cenário, senão subiria junto com a carga e esconderia a amplificação;
- um valor por combinação de parâmetros do gerador. Com o corpo lognormal e o
  corte da spliced fixos na grade, só o peso da cauda (xi) varia, então há um
  registro por xi: p90 e p95 ficam no corpo e são iguais para todos os xi, e só
  o p99 muda.

O arquivo `data/processed/limiares.json` guarda esses valores, os parâmetros que
os geraram e o SHA-256 da grade. A marcação (`marcacao.py`) lê os limiares
**daqui**, não os recalcula, de modo que o mesmo arquivo dá sempre a mesma
marcação; ao marcar, os valores são conferidos contra os do manifesto da
execução.

Regra de violação: `viola_pXX = latencia_conjunto_ms > limiar_pXX_ms`
(violação é passar do limiar; igual não é violação).

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.data.limiares --gerar
    python -m src.data.limiares --mostrar
"""

import argparse
import hashlib
import json
from pathlib import Path

from src.simulation.cenarios import CAMINHO_PADRAO, Cenario, carregar_grade
from src.simulation.manifesto import sha256_grade

RAIZ = Path(__file__).resolve().parents[2]
PASTA_PADRAO = RAIZ / "data" / "processed"
NOME_ARQUIVO = "limiares.json"
VERSAO = 1

# Percentil da chamada individual de cada limiar (p95 é a referência).
PERCENTIS = {"p90": 0.90, "p95": 0.95, "p99": 0.99}
REFERENCIA = "p95"
REGRA = "viola_pXX = latencia_conjunto_ms > limiar_pXX_ms"


def _registro(cenario: Cenario) -> dict:
    p = cenario.params
    return {
        "xi": cenario.xi,
        "u_ms": p.u,
        "beta_ms": p.beta,
        **{f"limiar_{nome}_ms": p.quantil(q) for nome, q in PERCENTIS.items()},
    }


def calcular_limiares(cenarios, caminho_grade: Path | str = CAMINHO_PADRAO) -> dict:
    """Limiares em ms, um registro por xi, a partir dos cenários da grade."""
    c0 = cenarios[0]
    for c in cenarios:
        if (c.mediana_ms, c.sigma, c.p_corte) != (c0.mediana_ms, c0.sigma, c0.p_corte):
            raise ValueError("a grade mistura corpos lognormais ou cortes diferentes")
    registros = {}
    for c in cenarios:
        registros.setdefault(c.xi, _registro(c))
    return {
        "versao": VERSAO,
        "regra": REGRA,
        "referencia": REFERENCIA,
        "percentis": dict(PERCENTIS),
        "grade": {"arquivo": Path(caminho_grade).name, "sha256": sha256_grade(caminho_grade)},
        "gerador": {
            "mediana_ms": c0.mediana_ms,
            "sigma": c0.sigma,
            "p_corte": c0.p_corte,
        },
        "limiares": [registros[xi] for xi in sorted(registros)],
    }


def gravar_limiares(limiares: dict, pasta: Path | str = PASTA_PADRAO) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / NOME_ARQUIVO
    texto = json.dumps(limiares, indent=2, ensure_ascii=False) + "\n"
    # floats em JSON usam repr: a releitura devolve exatamente os mesmos valores
    caminho.write_text(texto, encoding="utf-8", newline="\n")
    return caminho


def carregar_limiares(caminho: Path | str) -> dict:
    limiares = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if limiares.get("versao") != VERSAO:
        raise ValueError(f"versão de limiares {limiares.get('versao')!r}, esperada {VERSAO}")
    return limiares


def sha256_limiares(caminho: Path | str) -> str:
    """SHA-256 do arquivo (CRLF normalizado para LF)."""
    dados = Path(caminho).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(dados).hexdigest()


def limiares_do_cenario(limiares: dict, cenario: Cenario | dict) -> dict:
    """Devolve {'p90': ms, 'p95': ms, 'p99': ms} do xi do cenário.

    Confere também os parâmetros do gerador (mediana, sigma e corte) contra os
    do arquivo. Falha com ValueError se o xi não existir ou se o gerador diferir.
    """
    c = cenario if isinstance(cenario, dict) else {
        "xi": cenario.xi,
        "mediana_ms": cenario.mediana_ms,
        "sigma": cenario.sigma,
        "p_corte": cenario.p_corte,
    }
    g = limiares["gerador"]
    for chave in ("mediana_ms", "sigma", "p_corte"):
        if c[chave] != g[chave]:
            raise ValueError(
                f"{chave} do cenário ({c[chave]!r}) difere do arquivo de limiares ({g[chave]!r})"
            )
    for registro in limiares["limiares"]:
        if registro["xi"] == c["xi"]:
            return {nome: registro[f"limiar_{nome}_ms"] for nome in limiares["percentis"]}
    raise ValueError(f"xi = {c['xi']!r} não consta no arquivo de limiares")


def main() -> None:
    ap = argparse.ArgumentParser(description="Metadados dos limiares de SLA")
    acao = ap.add_mutually_exclusive_group(required=True)
    acao.add_argument("--gerar", action="store_true", help="calcula e grava limiares.json")
    acao.add_argument("--mostrar", action="store_true", help="imprime os limiares gravados")
    ap.add_argument("--grade", default=str(CAMINHO_PADRAO), help="caminho do TOML")
    ap.add_argument("--pasta", default=str(PASTA_PADRAO), help="pasta de saída/leitura")
    args = ap.parse_args()

    if args.gerar:
        limiares = calcular_limiares(carregar_grade(args.grade), args.grade)
        print(f"limiares gravados em {gravar_limiares(limiares, args.pasta)}")
    else:
        limiares = carregar_limiares(Path(args.pasta) / NOME_ARQUIVO)
    print(f"{'xi':>5} {'p90 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10}")
    for r in limiares["limiares"]:
        print(
            f"{r['xi']:>5.2f} {r['limiar_p90_ms']:>10.4f} "
            f"{r['limiar_p95_ms']:>10.4f} {r['limiar_p99_ms']:>10.4f}"
        )


if __name__ == "__main__":
    main()
