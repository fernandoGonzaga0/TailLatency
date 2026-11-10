"""Verificação das métricas por execução — Fase 6, item 1.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.analysis.teste_metricas

Se aparecer "Métricas funcionando corretamente.", as funções estão verificadas.
Usa só dados sintéticos pequenos; roda em menos de um segundo.
"""

import numpy as np

from src.analysis.metricas import (
    METRICAS,
    amplificacao,
    contagem_acima,
    contagens_violacao,
    fracao_acima,
    metricas_execucao,
    percentil,
)

LIMIARES = {"p90": 18.0, "p95": 22.0, "p99": 33.0}


def _tabela(lat, serv=None, espera=None):
    lat = np.asarray(lat, dtype=float)
    return {
        "latencia_conjunto_ms": lat,
        "servico_conjunto_ms": lat if serv is None else np.asarray(serv, dtype=float),
        "espera_media_ms": np.zeros_like(lat) if espera is None else np.asarray(espera, dtype=float),
    }


def teste_percentis_valores_conhecidos():
    x = np.arange(1, 101)  # 1..100, interpolação linear
    assert percentil(x, 50) == 50.5
    assert abs(percentil(x, 99) - 99.01) < 1e-12
    assert abs(amplificacao(x) - 99.01 / 50.5) < 1e-12
    assert percentil(x, 0) == 1 and percentil(x, 100) == 100


def teste_mesmo_metodo_do_numpy_do_manifesto():
    rng = np.random.default_rng(7)
    x = rng.lognormal(2.3, 0.5, 10_000)
    assert percentil(x, 99) == float(np.percentile(x, 99))
    assert amplificacao(x) == float(np.percentile(x, 99)) / float(np.percentile(x, 50))


def teste_violacao_estritamente_acima():
    x = [1.0, 2.0, 3.0, 3.0, 4.0]
    assert contagem_acima(x, 3.0) == 1  # igual ao limiar não é violação
    assert fracao_acima(x, 3.0) == 0.2
    assert fracao_acima(x, 0.0) == 1.0
    assert fracao_acima(x, 4.0) == 0.0


def teste_contagens_e_ordem_dos_limiares():
    rng = np.random.default_rng(1)
    x = rng.lognormal(2.3, 0.5, 5_000)
    c = contagens_violacao(x, LIMIARES)
    assert c["p99"] <= c["p95"] <= c["p90"]
    assert all(isinstance(v, int) for v in c.values())
    try:
        contagens_violacao(x, {"p90": 30.0, "p95": 22.0, "p99": 33.0})
    except ValueError:
        pass
    else:
        raise AssertionError("limiares fora de ordem deveriam falhar")
    try:
        contagens_violacao(x, {"p90": 18.0, "p95": 22.0})
    except ValueError:
        pass
    else:
        raise AssertionError("limiar faltando deveria falhar")


def teste_metricas_execucao_chaves_e_valores():
    lat = np.array([10.0, 15.0, 20.0, 25.0, 40.0] * 200)
    espera = np.array([0.0, 0.0, 1.0, 2.0, 5.0] * 200)
    m = metricas_execucao(_tabela(lat, espera=espera), LIMIARES)
    assert m["n_requisicoes"] == 1000
    assert set(METRICAS) <= set(m)
    assert m["p50_ms"] == float(np.percentile(lat, 50))
    assert m["p99_ms"] == float(np.percentile(lat, 99))
    assert m["amplificacao_p99_p50"] == m["p99_ms"] / m["p50_ms"]
    assert m["latencia_media_ms"] == float(np.mean(lat))
    assert m["espera_media_chamada_ms"] == float(np.mean(espera))
    # 25 e 40 passam de 22 (p95) e só 40 passa de 33 (p99); só 25 e 40 passam de 18? 20, 25 e 40
    assert m["violacao_p90"] == 3 / 5
    assert m["violacao_p95"] == 2 / 5
    assert m["violacao_p99"] == 1 / 5


def teste_violacao_do_servico_separa_a_fila():
    lat = np.array([10.0, 30.0, 50.0, 50.0])
    serv = np.array([10.0, 10.0, 10.0, 50.0])  # só a última viola por causa do serviço
    m = metricas_execucao(_tabela(lat, serv=serv), LIMIARES)
    assert m["violacao_p95"] == 3 / 4
    assert m["violacao_servico_p95"] == 1 / 4
    for n in ("p90", "p95", "p99"):
        assert m[f"violacao_servico_{n}"] <= m[f"violacao_{n}"]  # o serviço é parte da latência


def teste_validacao_de_entradas():
    for tabela, limiares in [
        ({"latencia_conjunto_ms": [1.0]}, LIMIARES),  # colunas ausentes
        (_tabela([]), LIMIARES),  # vazia
        (_tabela([0.0, 0.0]), LIMIARES),  # p50 nulo
        ({**_tabela([1.0, 2.0]), "servico_conjunto_ms": np.array([1.0])}, LIMIARES),  # tamanhos
    ]:
        try:
            metricas_execucao(tabela, limiares)
        except ValueError:
            continue
        raise AssertionError("entrada inválida deveria falhar")
    for p in (-1, 101):
        try:
            percentil([1.0], p)
        except ValueError:
            continue
        raise AssertionError("p fora de 0..100 deveria falhar")


def main() -> None:
    testes = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"ok  {t.__name__}")
    print("Métricas funcionando corretamente.")


if __name__ == "__main__":
    main()
