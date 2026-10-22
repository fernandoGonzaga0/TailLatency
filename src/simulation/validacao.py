"""Validação do módulo de simulação — Etapa 5, passo 8 (objetivo específico 4).

Antes de gerar os dados definitivos, confere se o simulador, **sem contenção**
(C <= capacidade, então ninguém espera na fila), reproduz a teoria.

O que se confere, em cada cenário sem contenção
------------------------------------------------
1. **Sem fila de fato:** a espera de toda chamada é zero.
2. **Chamadas individuais:** a fração de chamadas abaixo dos quantis p50, p95 e
   p99 teóricos da latência individual é 0,50, 0,95 e 0,99.
3. **Probabilidade de violação** (primeira parte da hipótese): para cada
   limiar q em {0,90; 0,95; 0,99}, a fração de requisições cuja latência do
   conjunto passa do quantil q individual é 1 - q^N. Este é o critério central.
4. **Percentis do conjunto:** o máximo de N latências i.i.d. tem cdf F(x)^N,
   então o quantil p do conjunto é o quantil p^(1/N) da latência individual.
   Confere-se p50, p95 e p99 do conjunto, e o fator p99/p50 teórico é mostrado
   ao lado do simulado.

Critério de aceite (definido antes de rodar, sem afrouxar; seção 2.2)
-----------------------------------------------------------------------
O valor teórico deve estar dentro do intervalo de confiança de 95% da fração
simulada. Sem contenção as requisições são independentes (nenhuma chamada
espera por outra), então o IC é o de Wilson para uma proporção, sobre as
requisições de todas as replicações juntas. Cada verificação 2, 3 e 4 é um teste
a 95%, então em um conjunto grande de testes cerca de 5% falham só por acaso.
Por isso o procedimento foi fixado antes: um teste que falha é refeito **uma
vez** com outra seed_base (todas as replicações), sem mudar a tolerância; a
validação é aprovada se todos passam na primeira ou na repetição, e o relatório
mostra quantos precisaram da repetição (esperado: cerca de 5%). Falhar nas duas
é falha real e investiga-se o simulador.

O cenário (N, C, xi) decide só o tamanho do conjunto e o corpo da cauda: sem
contenção a violação não depende de xi (o limiar é um quantil da própria
distribuição), mas o gerador da cauda GPD é exercitado em cada xi.

Como usar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.validacao --rodar                 # grade sem contenção inteira (40 cenários)
    python -m src.simulation.validacao --rodar --xi 0.5 --processos 4
    python -m src.simulation.teste_validacao

O relatório vai para `results/validacao_modulo.json` e a tabela é impressa.
"""

import argparse
import dataclasses
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.simulation.cenarios import CAMINHO_PADRAO, Cenario, carregar_grade, rodar_replicacao

RAIZ = Path(__file__).resolve().parents[2]
SAIDA_PADRAO = RAIZ / "results" / "validacao_modulo.json"

Q_LIMIARES = (0.90, 0.95, 0.99)  # limiares de SLA (p90, p95, p99 da chamada individual)
P_PERCENTIS = (0.50, 0.95, 0.99)  # percentis conferidos
CONFIANCA = 0.95
SEED_BASE_REPETICAO = 7_000_000  # somado à seed_base original na repetição


def wilson(k: int, n: int, confianca: float = CONFIANCA) -> tuple[float, float]:
    """IC de Wilson para uma proporção (k sucessos em n)."""
    z = float(norm.ppf(0.5 + confianca / 2))
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centro - meia), min(1.0, centro + meia)


def teoria_violacao(q: float, n_fan_out: int) -> float:
    """P(latência do conjunto > quantil q individual) sem contenção: 1 - q^N."""
    return 1.0 - q**n_fan_out


def quantil_conjunto(cenario: Cenario, p: float) -> float:
    """Quantil p do máximo de N latências i.i.d.: quantil p^(1/N) da individual."""
    return float(cenario.params.quantil(p ** (1.0 / cenario.n_fan_out)))


def _teste(nome: str, k: int, n: int, teorico: float, referencia: str) -> dict:
    lo, hi = wilson(k, n)
    return {
        "teste": nome,
        "referencia": referencia,
        "n": n,
        "k": k,
        "simulado": k / n,
        "teorico": teorico,
        "ic95": [lo, hi],
        "passou": lo <= teorico <= hi,
    }


def contar_replicacao(cenario: Cenario, replicacao: int) -> dict:
    """Roda uma replicação e devolve só contagens e as latências do conjunto."""
    resultado = rodar_replicacao(cenario, replicacao)
    p = cenario.params
    n = len(resultado)
    lat = np.array([r.latencia_conjunto for r in resultado])
    chamadas = np.array([c.latencia for r in resultado for c in r.chamadas])
    espera_maxima = max(c.espera for r in resultado for c in r.chamadas)
    return {
        "n": n,
        "n_chamadas": int(chamadas.size),
        "espera_maxima": float(espera_maxima),
        "violacoes": {str(q): int(np.sum(lat > p.quantil(q))) for q in Q_LIMIARES},
        "conjunto_abaixo": {
            str(pp): int(np.sum(lat <= quantil_conjunto(cenario, pp))) for pp in P_PERCENTIS
        },
        "chamada_abaixo": {
            str(pp): int(np.sum(chamadas <= p.quantil(pp))) for pp in P_PERCENTIS
        },
        "latencias": lat.tolist(),
    }


def _contar(args):
    return contar_replicacao(*args)


def agregar(cenario: Cenario, contagens: list[dict]) -> dict:
    """Junta as replicações de um cenário e monta os testes e o resumo."""
    n = sum(c["n"] for c in contagens)
    n_ch = sum(c["n_chamadas"] for c in contagens)
    testes = []
    for q in Q_LIMIARES:
        k = sum(c["violacoes"][str(q)] for c in contagens)
        testes.append(
            _teste(
                f"violacao_p{round(100 * q)}",
                k, n, teoria_violacao(q, cenario.n_fan_out), f"1 - {q}^{cenario.n_fan_out}",
            )
        )  # fmt: skip
    for pp in P_PERCENTIS:
        k = sum(c["conjunto_abaixo"][str(pp)] for c in contagens)
        testes.append(_teste(f"conjunto_abaixo_p{round(100 * pp)}", k, n, pp, "p do conjunto"))
    for pp in P_PERCENTIS:
        k = sum(c["chamada_abaixo"][str(pp)] for c in contagens)
        testes.append(_teste(f"chamada_abaixo_p{round(100 * pp)}", k, n_ch, pp, "p da chamada"))

    lat = np.concatenate([np.asarray(c["latencias"]) for c in contagens])
    p50, p99 = float(np.percentile(lat, 50)), float(np.percentile(lat, 99))
    t50, t99 = quantil_conjunto(cenario, 0.50), quantil_conjunto(cenario, 0.99)
    return {
        "id": cenario.id,
        "n_fan_out": cenario.n_fan_out,
        "concorrencia": cenario.concorrencia,
        "xi": cenario.xi,
        "replicacoes": len(contagens),
        "n_requisicoes": n,
        "sem_espera": all(c["espera_maxima"] == 0.0 for c in contagens),
        "p50_conjunto_ms": {"simulado": p50, "teorico": t50},
        "p99_conjunto_ms": {"simulado": p99, "teorico": t99},
        "amplificacao_p99_p50": {"simulado": p99 / p50, "teorico": t99 / t50},
        "testes": testes,
    }


def validar_cenario(cenario: Cenario, executor=None) -> dict:
    """Validação de um cenário sem contenção, com repetição única dos testes que falham."""
    if not cenario.sem_contencao:
        raise ValueError(f"{cenario.id} tem contenção; a validação é só sem contenção")
    mapa = executor.map if executor is not None else map

    def rodar(c: Cenario) -> list[dict]:
        return list(mapa(_contar, [(c, k) for k in range(c.n_replicacoes)]))

    primeira = agregar(cenario, rodar(cenario))
    resultado = dict(primeira)
    reprovados = [t["teste"] for t in primeira["testes"] if not t["passou"]]
    resultado["repetidos"] = reprovados
    if reprovados:
        outra = dataclasses.replace(cenario, seed_base=cenario.seed_base + SEED_BASE_REPETICAO)
        segunda = {t["teste"]: t for t in agregar(outra, rodar(outra))["testes"]}
        for t in resultado["testes"]:
            if t["teste"] in reprovados:
                t["repeticao"] = segunda[t["teste"]]
                t["aprovado_na_repeticao"] = segunda[t["teste"]]["passou"]
    for t in resultado["testes"]:
        t["aprovado"] = t["passou"] or t.get("aprovado_na_repeticao", False)
    resultado["aprovado"] = resultado["sem_espera"] and all(t["aprovado"] for t in resultado["testes"])
    return resultado


def resumo_geral(resultados: list[dict]) -> dict:
    testes = [t for r in resultados for t in r["testes"]]
    falhas_1a = sum(not t["passou"] for t in testes)
    return {
        "cenarios": len(resultados),
        "testes": len(testes),
        "falharam_na_primeira": falhas_1a,
        "fracao_falhas_primeira": falhas_1a / len(testes),
        "esperado_por_acaso": 1 - CONFIANCA,
        "reprovados_apos_repeticao": sum(not t["aprovado"] for t in testes),
        "cenarios_sem_espera": sum(r["sem_espera"] for r in resultados),
        "aprovado": all(r["aprovado"] for r in resultados),
    }


def imprimir(resultados: list[dict], resumo: dict) -> None:
    print(f"{'cenário':<18}{'viol p90':>18}{'viol p95':>18}{'viol p99':>18}  p99/p50 sim (teo)  repetidos  ok")
    for r in resultados:
        cel = []
        for t in r["testes"][:3]:
            cel.append(f"{t['simulado']:.4f}/{t['teorico']:.4f}{'*' if not t['passou'] else ' '}")
        a = r["amplificacao_p99_p50"]
        print(
            f"{r['id']:<18}{cel[0]:>18}{cel[1]:>18}{cel[2]:>18}"
            f"  {a['simulado']:.2f} ({a['teorico']:.2f})   {len(r['repetidos']):>9}  {'sim' if r['aprovado'] else 'NÃO'}"
        )
    print("(* = violação fora do IC de 95% na primeira execução; 'repetidos' conta os testes, dos 9 do cenário, refeitos com outra seed)")
    print(
        f"{resumo['testes']} testes em {resumo['cenarios']} cenários: "
        f"{resumo['falharam_na_primeira']} fora do IC na primeira execução "
        f"({100 * resumo['fracao_falhas_primeira']:.1f}%; esperado por acaso ≈ 5%), "
        f"{resumo['reprovados_apos_repeticao']} reprovados após a repetição."
    )
    print("VALIDAÇÃO APROVADA" if resumo["aprovado"] else "VALIDAÇÃO REPROVADA")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validação do módulo contra 1 - q^N")
    ap.add_argument("--rodar", action="store_true", required=True)
    ap.add_argument("--grade", default=str(CAMINHO_PADRAO))
    ap.add_argument("--xi", type=float, nargs="*", help="só estes xi (padrão: todos da grade)")
    ap.add_argument("--fan-out", type=int, nargs="*", help="só estes N")
    ap.add_argument("--processos", type=int, default=1)
    ap.add_argument("--saida", default=str(SAIDA_PADRAO))
    args = ap.parse_args()

    cenarios = [c for c in carregar_grade(args.grade) if c.sem_contencao]
    if args.xi:
        cenarios = [c for c in cenarios if c.xi in args.xi]
    if args.fan_out:
        cenarios = [c for c in cenarios if c.n_fan_out in args.fan_out]
    t0 = time.perf_counter()
    resultados = []
    executor = ProcessPoolExecutor(args.processos) if args.processos > 1 else None
    try:
        for i, c in enumerate(cenarios, 1):
            resultados.append(validar_cenario(c, executor))
            print(f"[{i}/{len(cenarios)}] {c.id} ({time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        if executor:
            executor.shutdown()
    resumo = resumo_geral(resultados)
    for r in resultados:
        for t in r["testes"]:
            t.pop("latencias", None)
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(
        json.dumps({"resumo": resumo, "cenarios": resultados}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    imprimir(resultados, resumo)
    print(f"relatório em {saida}")
    raise SystemExit(0 if resumo["aprovado"] else 1)


if __name__ == "__main__":
    main()
