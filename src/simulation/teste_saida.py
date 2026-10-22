"""Verificação do formato de saída dos dados brutos — Etapa 5, passo 7.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_saida

Se aparecer "Saída funcionando corretamente.", o passo 7 está verificado.
Usa uma grade pequena (300 requisições por replicação), então roda em segundos.
"""

import gzip
import tempfile
from pathlib import Path

import numpy as np

from src.simulation.cenarios import buscar_cenario, carregar_grade, rodar_replicacao
from src.simulation.manifesto import (
    carregar_manifesto,
    executar_e_registrar,
    verificar_manifesto,
)
from src.simulation.saida import (
    COLUNAS,
    conteudo_csv,
    gravar_requisicoes,
    ler_requisicoes,
    nome_dados,
    sha256_conteudo,
)

GRADE = """
[geral]
seed_base = 2026
capacidade = 4
n_requisicoes = 300
n_aquecimento = 20
n_replicacoes = 3
[latencia]
mediana_ms = 10.0
sigma = 0.5
p_corte = 0.95
[grade]
fan_out = [3]
concorrencia = [1, 6]
xi = [0.5]
"""


def _preparar(id_cenario: str = "N003_C06_xi0.50", replicacao: int = 1):
    pasta = Path(tempfile.mkdtemp())
    grade = pasta / "grade.toml"
    grade.write_text(GRADE, encoding="utf-8")
    c = buscar_cenario(carregar_grade(grade), id_cenario)
    return pasta, grade, c, rodar_replicacao(c, replicacao)


def teste_nome_e_colunas():
    assert nome_dados("N003_C06_xi0.50", 1) == "requisicoes_N003_C06_xi0.50_rep1.csv.gz"
    assert COLUNAS == (
        "id_requisicao", "laco", "inicio_ms", "fim_ms", "latencia_conjunto_ms",
        "servico_conjunto_ms", "espera_max_ms", "espera_media_ms",
    )  # fmt: skip


def teste_conteudo_csv_tem_cabecalho_e_uma_linha_por_requisicao():
    _, _, _, resultado = _preparar()
    texto = conteudo_csv(resultado).decode("utf-8")
    linhas = texto.split("\n")
    assert linhas[0] == ",".join(COLUNAS)
    assert linhas[-1] == "" and len(linhas) == 1 + 300 + 1
    assert "\r" not in texto
    assert all(len(l.split(",")) == len(COLUNAS) for l in linhas[:-1])


def teste_releitura_devolve_exatamente_os_mesmos_valores():
    pasta, _, _, resultado = _preparar()
    info = gravar_requisicoes(resultado, pasta / "d.csv.gz")
    t = ler_requisicoes(pasta / "d.csv.gz")
    assert list(t.columns) == list(COLUNAS) and len(t) == 300 == info["linhas"]
    assert t["latencia_conjunto_ms"].tolist() == [r.latencia_conjunto for r in resultado]
    assert t["inicio_ms"].tolist() == [float(r.inicio) for r in resultado]
    assert t["fim_ms"].tolist() == [float(r.fim) for r in resultado]
    assert t["servico_conjunto_ms"].tolist() == [r.servico_conjunto for r in resultado]
    assert t["id_requisicao"].tolist() == [r.id_requisicao for r in resultado]
    assert t["laco"].tolist() == [r.laco for r in resultado]
    esp_max = [max(c.espera for c in r.chamadas) for r in resultado]
    assert t["espera_max_ms"].tolist() == esp_max


def teste_relacoes_entre_colunas():
    _, _, _, resultado = _preparar()
    pasta = Path(tempfile.mkdtemp())
    gravar_requisicoes(resultado, pasta / "d.csv.gz")
    t = ler_requisicoes(pasta / "d.csv.gz")
    assert np.allclose(t["fim_ms"] - t["inicio_ms"], t["latencia_conjunto_ms"])
    assert (t["latencia_conjunto_ms"] >= t["servico_conjunto_ms"] - 1e-9).all()
    assert (t["espera_media_ms"] <= t["espera_max_ms"] + 1e-12).all()
    assert (t["espera_max_ms"] >= 0).all()


def teste_gravacao_deterministica_byte_a_byte():
    pasta, _, c, resultado = _preparar()
    a, b = pasta / "a.csv.gz", pasta / "b.csv.gz"
    ia = gravar_requisicoes(resultado, a)
    ib = gravar_requisicoes(rodar_replicacao(c, 1), b)
    assert a.read_bytes() == b.read_bytes()  # mtime=0 e sem nome no cabeçalho
    assert ia["sha256_conteudo"] == ib["sha256_conteudo"] == sha256_conteudo(a)
    assert gzip.decompress(a.read_bytes()) == conteudo_csv(resultado)


def teste_gravacao_atomica_sem_temporario_sobrando():
    pasta, _, _, resultado = _preparar()
    gravar_requisicoes(resultado, pasta / "sub" / "d.csv.gz")  # cria a pasta
    assert sorted(p.name for p in (pasta / "sub").iterdir()) == ["d.csv.gz"]


def teste_executar_e_registrar_grava_dados_e_manifesto():
    pasta, grade, c, _ = _preparar()
    saida = pasta / "raw"
    _, caminho = executar_e_registrar(c, 1, saida, grade)
    m = carregar_manifesto(caminho)
    assert m["versao_manifesto"] == 2
    d = m["dados"]
    assert d["arquivo"] == nome_dados(c.id, 1) and d["linhas"] == 300
    assert (saida / d["arquivo"]).exists()
    assert d["sha256_conteudo"] == sha256_conteudo(saida / d["arquivo"])
    v = verificar_manifesto(m, grade, saida)
    assert v["ok"] and v["diferencas"] == [], v


def teste_sem_dados_nao_cria_arquivo():
    pasta, grade, c, _ = _preparar()
    saida = pasta / "raw"
    _, caminho = executar_e_registrar(c, 1, saida, grade, gravar_dados=False)
    m = carregar_manifesto(caminho)
    assert m["dados"] is None
    assert [p.name for p in saida.iterdir()] == [caminho.name]
    assert verificar_manifesto(m, grade, saida)["ok"]


def _gerar_com_dados():
    pasta, grade, c, _ = _preparar()
    saida = pasta / "raw"
    _, caminho = executar_e_registrar(c, 1, saida, grade)
    return saida, grade, carregar_manifesto(caminho)


def teste_verificacao_detecta_arquivo_ausente():
    saida, grade, m = _gerar_com_dados()
    (saida / m["dados"]["arquivo"]).unlink()
    v = verificar_manifesto(m, grade, saida)
    assert not v["ok"] and any("ausente" in d for d in v["diferencas"])


def teste_verificacao_detecta_arquivo_adulterado():
    saida, grade, m = _gerar_com_dados()
    caminho = saida / m["dados"]["arquivo"]
    texto = gzip.decompress(caminho.read_bytes()).decode("utf-8").split("\n")
    campos = texto[10].split(",")
    campos[4] = repr(float(campos[4]) * 3)  # altera uma latência
    texto[10] = ",".join(campos)
    caminho.write_bytes(gzip.compress("\n".join(texto).encode("utf-8")))
    v = verificar_manifesto(m, grade, saida)
    assert not v["ok"] and any("SHA-256" in d for d in v["diferencas"])


def teste_verificacao_detecta_linhas_removidas():
    saida, grade, m = _gerar_com_dados()
    caminho = saida / m["dados"]["arquivo"]
    texto = gzip.decompress(caminho.read_bytes()).decode("utf-8").split("\n")
    caminho.write_bytes(gzip.compress("\n".join(texto[:-6] + [""]).encode("utf-8")))
    v = verificar_manifesto(m, grade, saida)
    assert not v["ok"] and any("linhas" in d for d in v["diferencas"])


def teste_resumo_do_manifesto_confere_com_o_arquivo():
    saida, grade, m = _gerar_com_dados()
    m2 = dict(m, resumo=dict(m["resumo"], p99_ms=m["resumo"]["p99_ms"] * 1.01))
    v = verificar_manifesto(m2, grade, saida)  # refeito e arquivo discordam do resumo
    assert not v["ok"]
    assert any("no arquivo" in d for d in v["diferencas"])
    assert any("refeito" in d for d in v["diferencas"])


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Saída funcionando corretamente.")
