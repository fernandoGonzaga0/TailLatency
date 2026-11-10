"""Verificação do índice das execuções — Etapa 6 (Fase 5 do checklist).

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.data.teste_indice

Se aparecer "Índice funcionando corretamente.", o índice está verificado.
Usa grades pequenas (300 requisições), então roda em segundos.
"""

import csv
import io
import tempfile
from pathlib import Path

from src.data.indice import (
    COLUNAS,
    construir_indice,
    conteudo_csv,
    gravar_indice,
    resumo_indice,
    verificar_indice,
)
from src.data.marcacao import marcar_tudo, nome_registro
from src.data.teste_marcacao import _preparar
from src.simulation.manifesto import carregar_manifesto


def _pronto():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    marcar_tudo(raw, processed, progresso=lambda _: None)
    return base, raw, processed, manifestos


def teste_linhas_e_colunas():
    base, raw, processed, manifestos = _pronto()
    linhas = construir_indice(raw, processed)
    assert len(linhas) == 2
    assert [x["id"] for x in linhas] == sorted(x["id"] for x in linhas)  # ordenado
    for linha in linhas:
        assert tuple(linha) == COLUNAS
    m = carregar_manifesto(raw / f"manifesto_{linhas[0]['id']}_rep0.json")
    x = linhas[0]
    assert (x["seed"], x["n_fan_out"], x["concorrencia"], x["xi"]) == (
        m["seed"], m["cenario"]["n_fan_out"], m["cenario"]["concorrencia"], m["cenario"]["xi"],
    )
    assert x["sha256_bruto"] == m["dados"]["sha256_conteudo"]
    assert x["n_requisicoes"] == m["dados"]["linhas"]
    assert x["limiar_p95_ms"] == m["derivados"]["limiar_p95_ms"]
    assert x["violacao_p95"] == m["resumo"]["violacao_p95"] or abs(
        x["violacao_p95"] - m["resumo"]["violacao_p95"]) <= 1 / x["n_requisicoes"]
    assert x["commit"] == m["git"]["commit"] and x["python"] == m["ambiente"]["python"]
    assert x["violacao_p99"] <= x["violacao_p95"] <= x["violacao_p90"]


def teste_csv_deterministico_e_relido():
    base, raw, processed, manifestos = _pronto()
    linhas = construir_indice(raw, processed)
    a = gravar_indice(linhas, base / "a.csv").read_bytes()
    b = gravar_indice(construir_indice(raw, processed), base / "b.csv").read_bytes()
    assert a == b  # mesmos dados, mesmo arquivo, byte a byte
    assert not list(base.glob("*.tmp")) and b"\r" not in a
    lidas = list(csv.DictReader(io.StringIO(a.decode("utf-8"), newline="")))
    assert len(lidas) == 2 and tuple(lidas[0]) == COLUNAS
    assert float(lidas[0]["violacao_p95"]) == linhas[0]["violacao_p95"]  # float sem perda
    assert lidas[0]["sha256_marcado"] == linhas[0]["sha256_marcado"]


def teste_verificacao_confere():
    base, raw, processed, manifestos = _pronto()
    caminho = gravar_indice(construir_indice(raw, processed), base / "indice.csv")
    assert verificar_indice(caminho, raw, processed) == {"ok": True, "diferencas": []}


def teste_verificacao_detecta_linha_alterada():
    base, raw, processed, manifestos = _pronto()
    caminho = gravar_indice(construir_indice(raw, processed), base / "indice.csv")
    linhas = list(csv.reader(io.StringIO(caminho.read_text(encoding="utf-8"), newline="")))
    coluna = COLUNAS.index("seed")
    linhas[1][coluna] = str(int(linhas[1][coluna]) + 1)  # uma seed alterada
    saida = io.StringIO(newline="")
    csv.writer(saida, lineterminator="\n").writerows(linhas)
    caminho.write_text(saida.getvalue(), encoding="utf-8", newline="")
    v = verificar_indice(caminho, raw, processed)
    assert not v["ok"] and any("difere em seed" in d for d in v["diferencas"])


def teste_verificacao_detecta_execucao_a_mais_e_a_menos():
    base, raw, processed, manifestos = _pronto()
    linhas = construir_indice(raw, processed)
    caminho = gravar_indice(linhas[:1], base / "indice.csv")  # índice sem uma execução
    v = verificar_indice(caminho, raw, processed)
    assert not v["ok"] and any("não consta no índice" in d for d in v["diferencas"])
    caminho = gravar_indice(linhas, base / "indice.csv")
    m = carregar_manifesto(manifestos[0])
    (raw / manifestos[0].name).unlink()  # índice com uma execução que sumiu dos dados
    (processed / nome_registro(m["id"], m["replicacao"])).unlink()
    v = verificar_indice(caminho, raw, processed)
    assert not v["ok"] and any("não existe nos dados" in d for d in v["diferencas"])


def teste_verificacao_detecta_marcacao_alterada():
    base, raw, processed, manifestos = _pronto()
    caminho = gravar_indice(construir_indice(raw, processed), base / "indice.csv")
    m = carregar_manifesto(manifestos[0])
    reg = processed / nome_registro(m["id"], m["replicacao"])
    reg.write_text(reg.read_text(encoding="utf-8").replace('"sha256_conteudo": "', '"sha256_conteudo": "0', 2),
                   encoding="utf-8")
    v = verificar_indice(caminho, raw, processed)
    assert not v["ok"]


def teste_execucao_sem_registro_de_marcacao_falha():
    base, grade, cenarios, raw, processed, manifestos = _preparar()
    try:
        construir_indice(raw, processed)  # nada foi marcado ainda
    except ValueError as e:
        assert "sem registro de marcação" in str(e)
    else:
        raise AssertionError("execução sem marcação deveria falhar")


def teste_pasta_vazia_falha():
    try:
        construir_indice(Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()))
    except ValueError as e:
        assert "nenhum manifesto" in str(e)
    else:
        raise AssertionError("pasta sem manifestos deveria falhar")


def teste_resumo_homogeneidade():
    base, raw, processed, manifestos = _pronto()
    linhas = construir_indice(raw, processed)
    r = resumo_indice(linhas)
    assert r["execucoes"] == 2 and r["cenarios"] == 2 and len(r["ambientes"]) == 1
    outras = [dict(linhas[0]), dict(linhas[1], commit="outro", python="3.13.0")]
    r2 = resumo_indice(outras)
    assert len(r2["commits"]) == 2 and len(r2["ambientes"]) == 2


def teste_conteudo_csv_booleano_e_nulo():
    linha = {c: "" for c in COLUNAS}
    linha.update(alteracoes_locais=False, commit=None, xi=0.5)
    texto = conteudo_csv([linha]).decode().split("\n")[1].split(",")
    assert texto[COLUNAS.index("alteracoes_locais")] == "false"
    assert texto[COLUNAS.index("commit")] == ""
    assert texto[COLUNAS.index("xi")] == "0.5"


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Índice funcionando corretamente.")
