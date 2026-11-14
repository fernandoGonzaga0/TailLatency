"""Verificação da conferência cruzada — Fase 6, item 2.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.teste_conferencia

Se aparecer "Conferência funcionando corretamente.", a conferência detecta as
divergências que deve detectar e aceita o que é só ruído numérico. Monta um
ambiente sintético completo (bruto, marcado, registro, índice e limiares) em pasta
temporária; roda em poucos segundos.
"""

import csv
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.carga import carregar_limiares, ler_indice, limiares_do_xi
from src.analysis.conferencia import (
    conferir_cobertura,
    conferir_execucao,
    conferir_tudo,
    nome_manifesto,
    nome_registro,
)
from src.analysis.metricas import NOMES_LIMIAR, contagens_violacao, metricas_execucao
from src.simulation.saida import nome_dados, sha256_conteudo

ID = "N005_C04_xi0.50"
LIM = {"p90": 18.0, "p95": 22.0, "p99": 36.0}
COLUNAS_INDICE = [
    "id", "n_fan_out", "concorrencia", "xi", "replicacao", "seed", "n_requisicoes",
    "sha256_bruto", "sha256_marcado", "limiar_p90_ms", "limiar_p95_ms", "limiar_p99_ms",
    "violacao_p90", "violacao_p95", "violacao_p99", "p50_ms", "p99_ms",
    "amplificacao_p99_p50", "latencia_media_ms", "espera_media_chamada_ms",
    "commit", "alteracoes_locais",
]


def _tabela(semente: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    lat = rng.lognormal(2.3, 0.5, n)
    lat[3] = LIM["p95"]  # exatamente no limiar: não é violação
    return pd.DataFrame({
        "id_requisicao": np.arange(n),
        "laco": np.zeros(n, dtype=int),
        "inicio_ms": np.arange(n, dtype=float),
        "fim_ms": np.arange(n, dtype=float) + lat,
        "latencia_conjunto_ms": lat,
        "servico_conjunto_ms": lat,
        "espera_max_ms": np.zeros(n),
        "espera_media_ms": np.zeros(n),
    })


def _gravar_gz(caminho: Path, tabela: pd.DataFrame) -> None:
    texto = tabela.to_csv(index=False, lineterminator="\n")
    with gzip.open(caminho, "wb") as f:
        f.write(texto.encode("utf-8"))


def _ambiente(n_reps: int = 2):
    """Pastas raw e processed completas e coerentes, limiares e índice (lista de dicts)."""
    base = Path(tempfile.mkdtemp(prefix="teste_conferencia_"))
    raw, proc = base / "raw", base / "processed"
    raw.mkdir()
    proc.mkdir()
    limiares_json = {
        "versao": 1,
        "regra": "viola_pXX = latencia_conjunto_ms > limiar_pXX_ms",
        "percentis": {"p90": 0.9, "p95": 0.95, "p99": 0.99},
        "limiares": [
            {"xi": 0.5, "limiar_p90_ms": LIM["p90"], "limiar_p95_ms": LIM["p95"],
             "limiar_p99_ms": LIM["p99"]},
        ],
    }
    (proc / "limiares.json").write_text(json.dumps(limiares_json), encoding="utf-8")
    limiares = carregar_limiares(proc / "limiares.json")

    linhas = []
    for rep in range(n_reps):
        bruto = _tabela(semente=rep)
        _gravar_gz(raw / nome_dados(ID, rep), bruto)
        (raw / nome_manifesto(ID, rep)).write_text("{}", encoding="utf-8")
        marcado = bruto.copy()
        for nome in NOMES_LIMIAR:
            marcado[f"viola_{nome}"] = (bruto["latencia_conjunto_ms"] > LIM[nome]).astype(int)
        _gravar_gz(proc / nome_dados(ID, rep), marcado)
        sha_bruto = sha256_conteudo(raw / nome_dados(ID, rep))
        sha_marcado = sha256_conteudo(proc / nome_dados(ID, rep))
        cont = contagens_violacao(bruto["latencia_conjunto_ms"], LIM)
        n = len(bruto)
        registro = {
            "versao_marcacao": 1, "id": ID, "replicacao": rep,
            "bruto": {"arquivo": nome_dados(ID, rep), "sha256_conteudo": sha_bruto},
            "marcado": {"arquivo": nome_dados(ID, rep), "linhas": n, "sha256_conteudo": sha_marcado},
            "limiares_ms": dict(LIM),
            "violacoes": cont,
            "fracao_violacao": {k: v / n for k, v in cont.items()},
        }
        (proc / nome_registro(ID, rep)).write_text(json.dumps(registro), encoding="utf-8")
        m = metricas_execucao(bruto, LIM)
        linhas.append({
            "id": ID, "n_fan_out": 5, "concorrencia": 4, "xi": 0.5, "replicacao": rep,
            "seed": 100 + rep, "n_requisicoes": n,
            "sha256_bruto": sha_bruto, "sha256_marcado": sha_marcado,
            "limiar_p90_ms": LIM["p90"], "limiar_p95_ms": LIM["p95"], "limiar_p99_ms": LIM["p99"],
            "violacao_p90": cont["p90"] / n, "violacao_p95": cont["p95"] / n,
            "violacao_p99": cont["p99"] / n,
            "p50_ms": m["p50_ms"], "p99_ms": m["p99_ms"],
            "amplificacao_p99_p50": m["amplificacao_p99_p50"],
            "latencia_media_ms": m["latencia_media_ms"],
            "espera_media_chamada_ms": m["espera_media_chamada_ms"],
            "commit": "abc", "alteracoes_locais": False,
        })
    return base, raw, proc, limiares, linhas


def _indice_relido(base: Path, linhas: list[dict]) -> list[dict]:
    caminho = base / "indice.csv"
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUNAS_INDICE)
        for x in linhas:
            w.writerow([
                repr(x[c]) if isinstance(x[c], float)
                else ("false" if x[c] is False else x[c]) for c in COLUNAS_INDICE
            ])
    return ler_indice(caminho)


def _confere(base, raw, proc, limiares, linhas, rep=0, edita_indice=None):
    idx = [dict(x) for x in linhas]
    if edita_indice:
        edita_indice(idx[rep])
    idx = _indice_relido(base, idx)
    return conferir_execucao(idx[rep], limiares, raw, proc)


def _reescrever_marcado(proc: Path, rep: int, altera) -> None:
    caminho = proc / nome_dados(ID, rep)
    tabela = pd.read_csv(caminho, float_precision="round_trip")
    altera(tabela)
    _gravar_gz(caminho, tabela)


def teste_ambiente_coerente_confere():
    base, raw, proc, limiares, linhas = _ambiente()
    for rep in (0, 1):
        assert _confere(base, raw, proc, limiares, linhas, rep) == []
    idx = _indice_relido(base, linhas)
    r = conferir_tudo(idx, limiares, raw, proc)
    assert r["execucoes"] == 2 and r["divergencias"] == {} and r["cobertura"] == []


def teste_regra_estritamente_acima():
    # a latência [3] é exatamente o limiar p95: o marcado correto a deixa em 0
    base, raw, proc, limiares, linhas = _ambiente()
    bruto = pd.read_csv(raw / nome_dados(ID, 0), compression="gzip")
    assert bruto.loc[3, "latencia_conjunto_ms"] == LIM["p95"]
    assert _confere(base, raw, proc, limiares, linhas) == []
    # marcação feita com >= em vez de > é detectada
    _reescrever_marcado(proc, 0, lambda t: t.__setitem__("viola_p95", (t["latencia_conjunto_ms"] >= LIM["p95"]).astype(int)))
    achados = _confere(base, raw, proc, limiares, linhas)
    assert any("viola_p95 difere do recalculado em 1 requisições" in a for a in achados), achados


def teste_detecta_coluna_marcada_alterada():
    base, raw, proc, limiares, linhas = _ambiente()
    def vira(t):
        t.loc[10, "viola_p99"] = 1 - t.loc[10, "viola_p99"]
    _reescrever_marcado(proc, 0, vira)
    achados = _confere(base, raw, proc, limiares, linhas)
    assert any(a.startswith("marcado: viola_p99 difere") for a in achados), achados
    assert not any("viola_p90" in a or "viola_p95" in a for a in achados)  # só a coluna alterada
    # a outra execução segue conferindo
    assert _confere(base, raw, proc, limiares, linhas, rep=1) == []


def teste_detecta_valor_fora_de_zero_e_um_e_latencia_diferente():
    base, raw, proc, limiares, linhas = _ambiente()
    _reescrever_marcado(proc, 0, lambda t: t.__setitem__("viola_p90", t["viola_p90"] * 2))
    assert any("fora de 0 e 1" in a for a in _confere(base, raw, proc, limiares, linhas))
    base, raw, proc, limiares, linhas = _ambiente()
    def muda_lat(t):
        t.loc[7, "latencia_conjunto_ms"] += 0.5
    _reescrever_marcado(proc, 0, muda_lat)
    assert any("latencia_conjunto_ms difere do bruto" in a for a in _confere(base, raw, proc, limiares, linhas))


def teste_detecta_registro_divergente():
    base, raw, proc, limiares, linhas = _ambiente()
    caminho = proc / nome_registro(ID, 0)
    reg = json.loads(caminho.read_text(encoding="utf-8"))
    reg["violacoes"]["p95"] += 1
    caminho.write_text(json.dumps(reg), encoding="utf-8")
    achados = _confere(base, raw, proc, limiares, linhas)
    assert any(a.startswith("registro: violações p95") for a in achados), achados

    base, raw, proc, limiares, linhas = _ambiente()
    caminho = proc / nome_registro(ID, 0)
    reg = json.loads(caminho.read_text(encoding="utf-8"))
    reg["limiares_ms"]["p99"] = 37.0
    caminho.write_text(json.dumps(reg), encoding="utf-8")
    assert any("registro: limiar p99" in a for a in _confere(base, raw, proc, limiares, linhas))

    base, raw, proc, limiares, linhas = _ambiente()
    (proc / nome_registro(ID, 0)).write_text("não é json", encoding="utf-8")
    assert any(a.startswith("registro: ilegível") for a in _confere(base, raw, proc, limiares, linhas))


def teste_detecta_indice_divergente_e_tolera_ruido_numerico():
    base, raw, proc, limiares, linhas = _ambiente()
    achados = _confere(base, raw, proc, limiares, linhas,
                       edita_indice=lambda x: x.__setitem__("violacao_p95", x["violacao_p95"] + 0.01))
    assert any(a.startswith("índice: violacao_p95") for a in achados), achados
    achados = _confere(base, raw, proc, limiares, linhas,
                       edita_indice=lambda x: x.__setitem__("p99_ms", x["p99_ms"] * (1 + 1e-6)))
    assert any(a.startswith("índice: p99_ms") for a in achados), achados
    achados = _confere(base, raw, proc, limiares, linhas,
                       edita_indice=lambda x: x.__setitem__("limiar_p95_ms", 23.0))
    assert any("índice: limiar_p95_ms" in a for a in achados), achados
    # ruído de 1e-13 relativo (versões de NumPy) é aceito
    ruido = lambda x: (x.__setitem__("p50_ms", x["p50_ms"] * (1 + 1e-13)),
                       x.__setitem__("latencia_media_ms", x["latencia_media_ms"] * (1 - 1e-13)))
    assert _confere(base, raw, proc, limiares, linhas, edita_indice=ruido) == []


def teste_detecta_arquivos_ausentes_e_bruto_adulterado():
    base, raw, proc, limiares, linhas = _ambiente()
    (proc / nome_dados(ID, 0)).unlink()
    assert any(a.startswith("marcado: arquivo ausente") for a in _confere(base, raw, proc, limiares, linhas))
    (proc / nome_registro(ID, 1)).unlink()
    assert any(a.startswith("registro: arquivo ausente") for a in _confere(base, raw, proc, limiares, linhas, rep=1))

    base, raw, proc, limiares, linhas = _ambiente()
    bruto = pd.read_csv(raw / nome_dados(ID, 0), float_precision="round_trip")
    bruto.loc[5, "latencia_conjunto_ms"] += 1.0
    _gravar_gz(raw / nome_dados(ID, 0), bruto)
    achados = _confere(base, raw, proc, limiares, linhas)
    assert len(achados) == 1 and achados[0].startswith("bruto:") and "SHA-256" in achados[0], achados
    (raw / nome_dados(ID, 1)).unlink()
    assert _confere(base, raw, proc, limiares, linhas, rep=1)[0].startswith("bruto:")


def teste_cobertura_falta_e_sobra():
    base, raw, proc, limiares, linhas = _ambiente()
    idx = _indice_relido(base, linhas)
    assert conferir_cobertura(idx, raw, proc) == []
    (raw / nome_manifesto(ID, 1)).unlink()
    (proc / "marcacao_OUTRO_rep0.json").write_text("{}", encoding="utf-8")
    (raw / "requisicoes_OUTRO_rep0.csv.gz").write_bytes(b"")
    achados = conferir_cobertura(idx, raw, proc)
    assert any("falta " + nome_manifesto(ID, 1) in a for a in achados), achados
    assert any("marcacao_OUTRO_rep0.json existe e não consta no índice" in a for a in achados)
    assert any("requisicoes_OUTRO_rep0.csv.gz existe e não consta no índice" in a for a in achados)
    # limiares.json não é arquivo de execução: não conta como sobra
    assert not any("limiares.json" in a for a in achados)


def teste_conferir_tudo_isola_falhas():
    base, raw, proc, limiares, linhas = _ambiente()
    _reescrever_marcado(proc, 0, lambda t: t.__setitem__("viola_p90", 1 - t["viola_p90"]))
    idx = _indice_relido(base, linhas)
    r = conferir_tudo(idx, limiares, raw, proc)
    assert set(r["divergencias"]) == {(ID, 0)}  # a execução 1 continua conferindo
    assert r["execucoes"] == 2


def main() -> None:
    testes = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"ok  {t.__name__}")
    print("Conferência funcionando corretamente.")


if __name__ == "__main__":
    main()
