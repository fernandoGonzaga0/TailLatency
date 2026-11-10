"""Verificação da marcação por limiar e dos metadados — Etapa 6 (Fase 5 do checklist).

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.data.teste_marcacao

Se aparecer "Marcação funcionando corretamente.", a marcação está verificada.
Usa grades pequenas, então roda em segundos (o teste contra a teoria usa 20.000
requisições e leva alguns segundos a mais).
"""

import gzip
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from src.data.limiares import (
    NOME_ARQUIVO,
    calcular_limiares,
    carregar_limiares,
    gravar_limiares,
    limiares_do_cenario,
)
from src.data.marcacao import (
    COLUNAS_VIOLACAO,
    marcar_conteudo,
    marcar_execucao,
    marcar_tudo,
    nome_registro,
    verificar_marcacao,
)
from src.simulation.cenarios import buscar_cenario, carregar_grade
from src.simulation.manifesto import carregar_manifesto, executar_e_registrar
from src.simulation.saida import COLUNAS, ler_requisicoes, nome_dados
from src.simulation.validacao import wilson

GRADE = """
[geral]
seed_base = 2026
capacidade = 4
n_requisicoes = {n}
n_aquecimento = 20
n_replicacoes = 3
[latencia]
mediana_ms = 10.0
sigma = 0.5
p_corte = 0.95
[grade]
fan_out = [3]
concorrencia = [1, 6]
xi = [0.5, 1.1]
"""


def _preparar(n: int = 300):
    """Gera dados brutos de 2 cenários em pastas temporárias e os limiares."""
    base = Path(tempfile.mkdtemp())
    grade = base / "grade.toml"
    grade.write_text(GRADE.format(n=n), encoding="utf-8")
    cenarios = carregar_grade(grade)
    raw, processed = base / "raw", base / "processed"
    manifestos = []
    for id_cenario in ("N003_C06_xi0.50", "N003_C01_xi1.10"):
        c = buscar_cenario(cenarios, id_cenario)
        _, m = executar_e_registrar(c, 0, raw, grade)
        manifestos.append(m)
    gravar_limiares(calcular_limiares(cenarios, grade), processed)
    return base, grade, cenarios, raw, processed, manifestos


def teste_limiares_conhecidos():
    grade = Path(tempfile.mkdtemp()) / "grade.toml"
    grade.write_text(GRADE.format(n=300).replace("[0.5, 1.1]", "[0.2, 0.5, 0.8, 1.1]"), encoding="utf-8")
    lim = calcular_limiares(carregar_grade(grade), grade)
    r = {x["xi"]: x for x in lim["limiares"]}
    assert sorted(r) == [0.2, 0.5, 0.8, 1.1]
    for x in r.values():  # p90 e p95 estão no corpo: iguais para todos os xi
        assert round(x["limiar_p90_ms"], 2) == 18.98 and round(x["limiar_p95_ms"], 2) == 22.76
    assert [round(r[xi]["limiar_p99_ms"], 2) for xi in (0.2, 0.5, 0.8, 1.1)] == [
        33.24, 36.40, 40.86, 47.20,
    ]
    assert len({x["limiar_p90_ms"] for x in r.values()}) == 1
    assert len({x["limiar_p95_ms"] for x in r.values()}) == 1
    assert lim["referencia"] == "p95" and lim["percentis"] == {"p90": 0.9, "p95": 0.95, "p99": 0.99}


def teste_limiares_gravar_carregar_e_buscar():
    base, grade, cenarios, raw, processed, _ = _preparar()
    lim = carregar_limiares(processed / NOME_ARQUIVO)
    assert lim == calcular_limiares(cenarios, grade)  # releitura sem perda
    c = buscar_cenario(cenarios, "N003_C06_xi0.50")
    assert limiares_do_cenario(lim, c)["p95"] == c.params.quantil(0.95)
    for chave, valor in (("xi", 0.7), ("sigma", 0.6)):
        outro = {"xi": c.xi, "mediana_ms": c.mediana_ms, "sigma": c.sigma, "p_corte": c.p_corte}
        outro[chave] = valor
        try:
            limiares_do_cenario(lim, outro)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{chave} diferente deveria falhar")


def teste_regra_estritamente_acima():
    bruto = (
        ",".join(COLUNAS) + "\n"
        "0,0,0.0,1.0,10.0,5.0,0.0,0.0\n"      # abaixo dos três
        "1,0,0.0,1.0,20.0,5.0,0.0,0.0\n"      # acima do p90 (18), abaixo do p95 (22)
        "2,0,0.0,1.0,22.0,5.0,0.0,0.0\n"      # exatamente no p95: não viola o p95
        "3,0,0.0,1.0,22.5,5.0,0.0,0.0\n"      # acima do p95, abaixo do p99 (30)
        "4,0,0.0,1.0,31.0,5.0,0.0,0.0\n"      # acima dos três
    ).encode()
    marcado, contagem = marcar_conteudo(bruto, {"p90": 18.0, "p95": 22.0, "p99": 30.0})
    linhas = marcado.decode().split("\n")
    assert linhas[0] == ",".join(COLUNAS + COLUNAS_VIOLACAO)
    assert [l.split(",")[-3:] for l in linhas[1:-1]] == [
        ["0", "0", "0"], ["1", "0", "0"], ["1", "0", "0"], ["1", "1", "0"], ["1", "1", "1"],
    ]
    assert contagem == {"p90": 4, "p95": 2, "p99": 1, "linhas": 5}
    # as colunas originais continuam idênticas, como texto
    assert [l.rsplit(",", 3)[0] for l in linhas[1:-1]] == bruto.decode().split("\n")[1:-1]


def teste_cabecalho_inesperado_falha():
    try:
        marcar_conteudo(b"a,b,c\n1,2,3\n", {"p90": 1.0, "p95": 2.0, "p99": 3.0})
    except ValueError:
        return
    raise AssertionError("cabeçalho inesperado deveria falhar")


def teste_marcacao_de_ponta_a_ponta():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    for caminho in manifestos:
        m = carregar_manifesto(caminho)
        registro = marcar_execucao(caminho, processed)
        assert registro.name == nome_registro(m["id"], m["replicacao"])
        r = json.loads(registro.read_text(encoding="utf-8"))
        tabela = ler_requisicoes(processed / nome_dados(m["id"], m["replicacao"]))
        bruta = ler_requisicoes(raw / m["dados"]["arquivo"])
        assert list(tabela.columns) == list(COLUNAS + COLUNAS_VIOLACAO)
        assert len(tabela) == m["dados"]["linhas"] == r["marcado"]["linhas"]
        # as colunas originais voltam exatamente iguais às do bruto
        assert tabela[list(COLUNAS)].equals(bruta)
        # a fração marcada em p95 é a do resumo do manifesto
        assert abs(tabela["viola_p95"].mean() - m["resumo"]["violacao_p95"]) <= 1 / len(tabela)
        # limiares gravados no registro = derivados do manifesto
        for nome in ("p90", "p95", "p99"):
            assert math.isclose(r["limiares_ms"][nome], m["derivados"][f"limiar_{nome}_ms"], rel_tol=1e-9)
        # p99 ⊂ p95 ⊂ p90 (limiares crescentes)
        assert (tabela["viola_p99"] <= tabela["viola_p95"]).all()
        assert (tabela["viola_p95"] <= tabela["viola_p90"]).all()
        # a marcação bate com a comparação direta feita pela análise
        lat = tabela["latencia_conjunto_ms"].to_numpy()
        for nome in ("p90", "p95", "p99"):
            assert (tabela[f"viola_{nome}"].to_numpy() == (lat > r["limiares_ms"][nome])).all()
        assert verificar_marcacao(registro, raw, processed / NOME_ARQUIVO)["ok"]


def teste_sem_contencao_bate_com_a_teoria():
    base, grade, cenarios, raw, processed, _ = _preparar(n=20000)
    c = buscar_cenario(cenarios, "N003_C01_xi1.10")
    m = carregar_manifesto(raw / f"manifesto_{c.id}_rep0.json")
    marcar_execucao(raw / f"manifesto_{c.id}_rep0.json", processed)
    tabela = ler_requisicoes(processed / nome_dados(c.id, 0))
    n = len(tabela)
    for nome, q in (("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        k = int(tabela[f"viola_{nome}"].sum())
        teoria = 1 - q**c.n_fan_out
        lo, hi = wilson(k, n)
        assert lo <= teoria <= hi, f"{nome}: teoria {teoria:.4f} fora de [{lo:.4f}, {hi:.4f}]"
    assert m["resumo"]["n_requisicoes"] == n


def teste_com_contencao_viola_mais_que_sem():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    fr = {}
    for caminho in manifestos:
        m = carregar_manifesto(caminho)
        r = json.loads(marcar_execucao(caminho, processed).read_text(encoding="utf-8"))
        fr[m["id"]] = r["fracao_violacao"]["p95"]
    assert fr["N003_C06_xi0.50"] > 1 - 0.95**3 > 0  # com fila, acima da teoria
    assert fr["N003_C06_xi0.50"] > fr["N003_C01_xi1.10"]


def teste_marcacao_deterministica_e_atomica():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    caminho = manifestos[0]
    marcar_execucao(caminho, processed)
    m = carregar_manifesto(caminho)
    arq = processed / nome_dados(m["id"], m["replicacao"])
    primeira = arq.read_bytes()
    marcar_execucao(caminho, processed)
    assert arq.read_bytes() == primeira  # mesma marcação, mesmo arquivo, byte a byte
    assert not list(processed.glob("*.tmp"))
    # o bruto não foi tocado
    assert (raw / m["dados"]["arquivo"]).exists()
    assert not any("viola" in l for l in gzip.decompress((raw / m["dados"]["arquivo"]).read_bytes()).decode().split("\n")[:1])


def teste_verificacao_detecta_adulteracao():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    registro = marcar_execucao(manifestos[0], processed)
    r = json.loads(registro.read_text(encoding="utf-8"))
    marcado = processed / r["marcado"]["arquivo"]
    original = marcado.read_bytes()

    # (1) uma marca trocada no arquivo marcado
    texto = gzip.decompress(original).decode().split("\n")
    campos = texto[5].split(",")
    campos[-2] = "0" if campos[-2] == "1" else "1"
    texto[5] = ",".join(campos)
    marcado.write_bytes(gzip.compress("\n".join(texto).encode()))
    v = verificar_marcacao(registro, raw)
    assert not v["ok"] and any("marcado" in d for d in v["diferencas"])
    marcado.write_bytes(original)
    assert verificar_marcacao(registro, raw)["ok"]

    # (2) arquivo marcado ausente
    marcado.unlink()
    v = verificar_marcacao(registro, raw)
    assert not v["ok"] and any("ausente" in d for d in v["diferencas"])
    marcado.write_bytes(original)

    # (3) bruto alterado
    bruto = raw / r["bruto"]["arquivo"]
    bruto_original = bruto.read_bytes()
    linhas = gzip.decompress(bruto_original).decode().split("\n")
    bruto.write_bytes(gzip.compress("\n".join(linhas[:-3] + [""]).encode()))
    v = verificar_marcacao(registro, raw)
    assert not v["ok"] and any("bruto" in d for d in v["diferencas"])
    bruto.write_bytes(bruto_original)
    assert verificar_marcacao(registro, raw)["ok"]


def teste_limiares_alterados_so_avisam_na_verificacao_e_falham_ao_marcar():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    registro = marcar_execucao(manifestos[0], processed)
    caminho_lim = processed / NOME_ARQUIVO
    lim = json.loads(caminho_lim.read_text(encoding="utf-8"))
    for x in lim["limiares"]:
        x["limiar_p95_ms"] *= 1.05
    caminho_lim.write_text(json.dumps(lim, indent=2) + "\n", encoding="utf-8")
    # a verificação usa os limiares gravados no registro: confere, com aviso
    v = verificar_marcacao(registro, raw, caminho_lim)
    assert v["ok"] and any("limiares" in a for a in v["avisos"])
    # marcar de novo com limiares que divergem do manifesto falha
    try:
        marcar_execucao(manifestos[0], processed)
    except ValueError as e:
        assert "difere do manifesto" in str(e)
    else:
        raise AssertionError("limiares divergentes do manifesto deveriam falhar")


def teste_bruto_adulterado_nao_e_marcado():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    m = carregar_manifesto(manifestos[0])
    bruto = raw / m["dados"]["arquivo"]
    linhas = gzip.decompress(bruto.read_bytes()).decode().split("\n")
    bruto.write_bytes(gzip.compress("\n".join(linhas[:-2] + [""]).encode()))
    try:
        marcar_execucao(manifestos[0], processed)
    except ValueError as e:
        assert "SHA-256" in str(e)
    else:
        raise AssertionError("bruto adulterado deveria falhar")
    assert not list(processed.glob("requisicoes_*"))


def teste_marcar_tudo_e_retomada():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    r1 = marcar_tudo(raw, processed, progresso=lambda _: None)
    assert (r1["marcadas"], r1["puladas"], r1["falhas"]) == (2, 0, [])
    r2 = marcar_tudo(raw, processed, progresso=lambda _: None)
    assert (r2["marcadas"], r2["puladas"], r2["falhas"]) == (0, 2, [])  # retomável
    # apagar um marcado: só ele é refeito
    m = carregar_manifesto(manifestos[0])
    (processed / nome_dados(m["id"], m["replicacao"])).unlink()
    r3 = marcar_tudo(raw, processed, progresso=lambda _: None)
    assert (r3["marcadas"], r3["puladas"]) == (1, 1)
    # uma execução com problema aparece nas falhas e não derruba as outras
    bruto = raw / carregar_manifesto(manifestos[1])["dados"]["arquivo"]
    bruto.write_bytes(gzip.compress(b"lixo\n"))
    (processed / nome_registro(*[carregar_manifesto(manifestos[1])[k] for k in ("id", "replicacao")])).unlink()
    r4 = marcar_tudo(raw, processed, progresso=lambda _: None)
    assert len(r4["falhas"]) == 1 and r4["puladas"] == 1


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Marcação funcionando corretamente.")
