"""Verificação do manifesto de reprodutibilidade — Etapa 5, passo 6.

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_manifesto

Se aparecer "Manifesto funcionando corretamente.", o passo 6 está verificado.
Usa uma grade pequena (300 requisições por replicação), então roda em segundos.
"""

import copy
import json
import tempfile
from pathlib import Path

from src.simulation.cenarios import buscar_cenario, carregar_grade, rodar_replicacao
from src.simulation.manifesto import (
    _info_git,
    carregar_manifesto,
    executar_e_registrar,
    gerar_manifesto,
    gravar_manifesto,
    nome_manifesto,
    sha256_grade,
    verificar_manifesto,
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


def _grade_temporaria(texto: str = GRADE) -> Path:
    p = Path(tempfile.mkdtemp()) / "grade.toml"
    p.write_text(texto, encoding="utf-8")
    return p


def _preparar():
    grade = _grade_temporaria()
    c = buscar_cenario(carregar_grade(grade), "N003_C06_xi0.50")
    resultado = rodar_replicacao(c, 1)
    return grade, c, resultado, gerar_manifesto(c, 1, resultado, 0.5, grade)


def teste_manifesto_tem_o_conteudo_esperado():
    grade, c, _, m = _preparar()
    assert m["id"] == "N003_C06_xi0.50" and m["replicacao"] == 1
    assert m["seed"] == c.seed(1)
    assert m["cenario"]["n_fan_out"] == 3 and m["cenario"]["concorrencia"] == 6
    assert m["cenario"]["capacidade"] == 4 and m["cenario"]["seed_base"] == 2026
    assert abs(m["derivados"]["limiar_p95_ms"] - 22.7602) < 1e-3
    assert m["derivados"]["limiar_p90_ms"] < m["derivados"]["limiar_p95_ms"]
    assert m["derivados"]["limiar_p95_ms"] < m["derivados"]["limiar_p99_ms"]
    assert m["grade"]["sha256"] == sha256_grade(grade)
    for pacote in ("python", "numpy", "scipy", "simpy"):
        assert m["ambiente"][pacote]
    assert m["resumo"]["n_requisicoes"] == 300
    assert m["execucao"]["duracao_s"] == 0.5
    json.dumps(m)  # tem de ser serializável


def teste_hash_da_grade_ignora_quebra_de_linha():
    a = _grade_temporaria(GRADE)
    b = Path(tempfile.mkdtemp()) / "grade.toml"
    b.write_bytes(GRADE.replace("\n", "\r\n").encode("utf-8"))
    assert sha256_grade(a) == sha256_grade(b)
    c = _grade_temporaria(GRADE.replace("seed_base = 2026", "seed_base = 2027"))
    assert sha256_grade(a) != sha256_grade(c)


def teste_gravar_e_carregar_sem_perda():
    _, c, _, m = _preparar()
    pasta = Path(tempfile.mkdtemp()) / "sub" / "raw"  # a pasta é criada
    caminho = gravar_manifesto(m, pasta)
    assert caminho.name == nome_manifesto(c, 1) == "manifesto_N003_C06_xi0.50_rep1.json"
    assert carregar_manifesto(caminho) == m


def teste_verificacao_confere_com_execucao_recente():
    grade, _, _, m = _preparar()
    v = verificar_manifesto(m, grade)
    assert v["ok"] and v["diferencas"] == [] and v["avisos"] == []


def teste_verificacao_tolera_ruido_numerico_minusculo():
    _, _, _, m = _preparar()
    m2 = copy.deepcopy(m)
    m2["resumo"]["p99_ms"] *= 1 + 1e-12
    m2["resumo"]["latencia_media_ms"] *= 1 - 1e-12
    assert verificar_manifesto(m2)["ok"]


def teste_verificacao_detecta_resumo_adulterado():
    _, _, _, m = _preparar()
    for chave, fator in (("p99_ms", 1.01), ("espera_media_chamada_ms", 1.01), ("p50_ms", 0.99)):
        m2 = copy.deepcopy(m)
        m2["resumo"][chave] *= fator
        v = verificar_manifesto(m2)
        assert not v["ok"] and any(chave in d for d in v["diferencas"]), chave
    m3 = copy.deepcopy(m)
    m3["resumo"]["violacao_p95"] += 5 / 300  # 5 requisições de diferença
    assert not verificar_manifesto(m3)["ok"]
    m4 = copy.deepcopy(m)
    m4["resumo"]["violacao_p95"] += 1 / 300  # 1 requisição é tolerada
    assert verificar_manifesto(m4)["ok"]


def teste_verificacao_detecta_seed_e_cenario_alterados():
    _, _, _, m = _preparar()
    m2 = copy.deepcopy(m)
    m2["seed"] += 1
    v = verificar_manifesto(m2)
    assert not v["ok"] and any("seed" in d for d in v["diferencas"])
    m3 = copy.deepcopy(m)
    m3["cenario"]["n_fan_out"] = 5  # outro cenário, mesmo resumo gravado
    assert not verificar_manifesto(m3)["ok"]


def teste_mudanca_de_grade_e_de_ambiente_so_avisam():
    grade, _, _, m = _preparar()
    outra = _grade_temporaria(GRADE.replace("n_aquecimento = 20", "n_aquecimento = 30"))
    v = verificar_manifesto(m, outra)
    assert v["ok"] and any("grade" in a for a in v["avisos"])
    m2 = copy.deepcopy(m)
    m2["ambiente"]["numpy"] = "0.0.0"
    v2 = verificar_manifesto(m2, grade)
    assert v2["ok"] and any("numpy" in a for a in v2["avisos"])


def teste_executar_e_registrar():
    grade = _grade_temporaria()
    c = buscar_cenario(carregar_grade(grade), "N003_C01_xi0.50")
    pasta = Path(tempfile.mkdtemp())
    resultado, caminho = executar_e_registrar(c, 0, pasta, grade)
    esperado = rodar_replicacao(c, 0)
    assert [r.latencia_conjunto for r in resultado] == [r.latencia_conjunto for r in esperado]
    assert caminho.exists() and caminho.parent == pasta
    m = carregar_manifesto(caminho)
    assert m["id"] == c.id and m["replicacao"] == 0 and m["execucao"]["duracao_s"] >= 0
    assert verificar_manifesto(m, grade)["ok"]


def teste_git_fora_de_repositorio_nao_falha():
    info = _info_git(Path(tempfile.mkdtemp()))
    assert info == {"commit": None, "alteracoes_locais": None}


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Manifesto funcionando corretamente.")
