"""Verificação da execução da grade — Etapa 6 (Fase 5 do checklist).

Como rodar (venv ativado, a partir da raiz TCC1):

    python -m src.simulation.teste_executar_grade

Se aparecer "Execução da grade funcionando corretamente.", o script está verificado.
Usa uma grade pequena (2 cenários x 3 replicações de 300 requisições), então roda
em segundos.
"""

import gzip
import json
import tempfile
from pathlib import Path

from src.simulation import executar_grade as eg
from src.simulation.cenarios import carregar_grade
from src.simulation.manifesto import carregar_manifesto, nome_manifesto
from src.simulation.saida import nome_dados

GRADE = """
[geral]
seed_base = 2026
capacidade = 4
n_requisicoes = 300
n_aquecimento = 20
n_replicacoes = {reps}
[latencia]
mediana_ms = 10.0
sigma = 0.5
p_corte = 0.95
[grade]
fan_out = [3]
concorrencia = [1, 6]
xi = [0.5]
"""


def _preparar(reps: int = 3):
    base = Path(tempfile.mkdtemp())
    grade = base / "grade.toml"
    grade.write_text(GRADE.format(reps=reps), encoding="utf-8")
    cenarios = carregar_grade(grade)
    return base, grade, cenarios, base / "raw"


def _silencio(_):
    pass


def teste_planejar_filtros_e_ordem():
    base, grade, cenarios, raw = _preparar()
    todas = eg.planejar(cenarios)
    assert len(todas) == 6
    assert [t[0].concorrencia for t in todas][:3] == [6, 6, 6]  # mais pesadas primeiro
    assert len(eg.planejar(cenarios, ids=["N003_C06_xi0.50"])) == 3
    assert [k for _, k in eg.planejar(cenarios, replicacoes=[1])] == [1, 1]
    for ids, reps in ((["N999_C01_xi0.50"], None), (None, [3]), (None, [-1])):
        try:
            eg.planejar(cenarios, ids, reps)
        except ValueError:
            continue
        raise AssertionError(f"filtro inválido deveria falhar: {ids} {reps}")


def teste_executa_tudo_e_os_dados_conferem():
    base, grade, cenarios, raw = _preparar()
    tarefas = eg.planejar(cenarios)
    r = eg.executar(tarefas, raw, grade, processos=1, progresso=_silencio)
    assert (r["executadas"], r["puladas"], r["falhas"]) == (6, 0, [])
    assert len(list(raw.glob("manifesto_*.json"))) == 6
    assert len(list(raw.glob("requisicoes_*.csv.gz"))) == 6
    assert not list(raw.glob("*.tmp"))
    for c, k in tarefas:
        assert eg.execucao_conferida(c, k, raw)


def teste_paralelo_igual_ao_serial():
    base1, grade1, cenarios1, raw1 = _preparar()
    base2, grade2, cenarios2, raw2 = _preparar()
    eg.executar(eg.planejar(cenarios1), raw1, grade1, processos=1, progresso=_silencio)
    r = eg.executar(eg.planejar(cenarios2), raw2, grade2, processos=2, progresso=_silencio)
    assert (r["executadas"], r["falhas"]) == (6, [])
    for c, k in eg.planejar(cenarios1):
        a = carregar_manifesto(raw1 / nome_manifesto(c, k))["dados"]["sha256_conteudo"]
        b = carregar_manifesto(raw2 / nome_manifesto(c, k))["dados"]["sha256_conteudo"]
        assert a == b, f"{c.id} rep{k}: o resultado depende do número de processos"


def teste_retomada_pula_o_que_confere():
    base, grade, cenarios, raw = _preparar()
    tarefas = eg.planejar(cenarios)
    eg.executar(tarefas, raw, grade, progresso=_silencio)
    r = eg.executar(tarefas, raw, grade, progresso=_silencio)
    assert (r["executadas"], r["puladas"]) == (0, 6)


def teste_retomada_refaz_o_que_nao_confere():
    base, grade, cenarios, raw = _preparar()
    tarefas = eg.planejar(cenarios)
    eg.executar(tarefas, raw, grade, progresso=_silencio)
    c0, c1, c2, c3 = tarefas[0][0], tarefas[1][0], tarefas[2][0], tarefas[3][0]
    k0, k1, k2, k3 = tarefas[0][1], tarefas[1][1], tarefas[2][1], tarefas[3][1]
    # (1) dado ausente
    (raw / nome_dados(c0.id, k0)).unlink()
    # (2) manifesto ilegível (escrita interrompida)
    (raw / nome_manifesto(c1, k1)).write_text('{"id": "N003', encoding="utf-8")
    # (3) dado adulterado
    dado = raw / nome_dados(c2.id, k2)
    linhas = gzip.decompress(dado.read_bytes()).decode().split("\n")
    dado.write_bytes(gzip.compress("\n".join(linhas[:-4] + [""]).encode()))
    # (4) manifesto sem a execução gravada (só o dado ficou)
    (raw / nome_manifesto(c3, k3)).unlink()
    assert not any(eg.execucao_conferida(c, k, raw) for c, k in tarefas[:4])
    assert all(eg.execucao_conferida(c, k, raw) for c, k in tarefas[4:])
    r = eg.executar(tarefas, raw, grade, progresso=_silencio)
    assert (r["executadas"], r["puladas"], r["falhas"]) == (4, 2, [])
    assert all(eg.execucao_conferida(c, k, raw) for c, k in tarefas)


def teste_seed_ou_cenario_diferente_nao_conta_como_conferido():
    base, grade, cenarios, raw = _preparar()
    tarefas = eg.planejar(cenarios)
    eg.executar(tarefas, raw, grade, progresso=_silencio)
    c, k = tarefas[0]
    caminho = raw / nome_manifesto(c, k)
    m = json.loads(caminho.read_text(encoding="utf-8"))
    m["seed"] += 1
    caminho.write_text(json.dumps(m), encoding="utf-8")
    assert not eg.execucao_conferida(c, k, raw)
    m["seed"] -= 1
    m["cenario"]["sigma"] = 0.6  # outra distribuição: não vale para a grade atual
    caminho.write_text(json.dumps(m), encoding="utf-8")
    assert not eg.execucao_conferida(c, k, raw)


def teste_aumentar_replicacoes_mantem_as_existentes():
    base, grade, cenarios, raw = _preparar(reps=3)
    eg.executar(eg.planejar(cenarios), raw, grade, progresso=_silencio)
    grade.write_text(GRADE.format(reps=4), encoding="utf-8")  # 3 -> 4 replicações
    novos = carregar_grade(grade)
    r = eg.executar(eg.planejar(novos), raw, grade, progresso=_silencio)
    assert (r["executadas"], r["puladas"]) == (2, 6)  # só a replicação 3 de cada cenário


def teste_falha_isolada_nao_derruba_as_outras():
    base, grade, cenarios, raw = _preparar()
    tarefas = eg.planejar(cenarios)
    original = eg._executar_tarefa

    def com_defeito(cenario, replicacao, pasta, grade_):
        if cenario.concorrencia == 6 and replicacao == 1:
            raise RuntimeError("defeito simulado")
        return original(cenario, replicacao, pasta, grade_)

    eg._executar_tarefa = com_defeito
    try:
        r = eg.executar(tarefas, raw, grade, processos=1, progresso=_silencio)
    finally:
        eg._executar_tarefa = original
    assert r["executadas"] == 5 and r["puladas"] == 0
    assert len(r["falhas"]) == 1 and r["falhas"][0][:2] == ("N003_C06_xi0.50", 1)
    assert "defeito simulado" in r["falhas"][0][2]
    # na próxima chamada só a que falhou é refeita
    r2 = eg.executar(tarefas, raw, grade, progresso=_silencio)
    assert (r2["executadas"], r2["puladas"], r2["falhas"]) == (1, 5, [])


def teste_arvore_do_git():
    limpa = {"commit": "abc", "alteracoes_locais": False}
    suja = {"commit": "abc", "alteracoes_locais": True}
    sem_git = {"commit": None, "alteracoes_locais": None}
    assert eg.checar_git(limpa) == (True, None)
    pode, msg = eg.checar_git(suja)
    assert not pode and "commit" in msg
    pode, msg = eg.checar_git(suja, permitir_suja=True)
    assert pode and msg.startswith("aviso")
    pode, msg = eg.checar_git(sem_git)
    assert pode and msg.startswith("aviso")


def teste_verificar_existentes():
    base, grade, cenarios, raw = _preparar()
    tarefas = eg.planejar(cenarios)
    r = eg.verificar_existentes(tarefas, raw)
    assert (r["conferidas"], len(r["ausentes"]), r["problemas"]) == (0, 6, [])
    eg.executar(tarefas[:4], raw, grade, progresso=_silencio)
    (raw / nome_dados(tarefas[0][0].id, tarefas[0][1])).unlink()
    r = eg.verificar_existentes(tarefas, raw)
    assert r["conferidas"] == 3 and len(r["ausentes"]) == 2
    assert r["problemas"] == [(tarefas[0][0].id, tarefas[0][1])]


def teste_progresso_informa_contagem_e_estimativa():
    base, grade, cenarios, raw = _preparar()
    linhas = []
    eg.executar(eg.planejar(cenarios)[:2], raw, grade, progresso=linhas.append)
    assert "2 a rodar" in linhas[0]
    assert linhas[1].startswith("[1/2]") and linhas[2].startswith("[2/2]")
    assert "faltam cerca de" in linhas[1]


def teste_formatar_tempo():
    assert eg._formatar(45) == "45s"
    assert eg._formatar(125) == "2min05s"
    assert eg._formatar(3725) == "1h02min"


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("teste_")]
    for nome, fn in testes:
        fn()
        print(f"ok  {nome}")
    print("Execução da grade funcionando corretamente.")
