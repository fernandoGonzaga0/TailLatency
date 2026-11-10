"""Execução da grade de cenários — Etapa 6 (Fase 5 do checklist).

Roda todas as execuções (cenário x replicação) de `config/grade.toml` com **um
comando**, em paralelo e de forma **retomável**, gravando em `data/raw/` o dado
bruto (`requisicoes_<id>_rep<k>.csv.gz`) e o manifesto
(`manifesto_<id>_rep<k>.json`) de cada uma, pelo mesmo caminho do passo 6/7 da
Fase 4 (`manifesto.executar_e_registrar`). Com a grade oficial são 120 cenários
x 5 replicações = 600 execuções (cerca de 2 h em um núcleo; com 4 processos,
cerca de meia hora).

Decisões de projeto
-------------------
- **Paralelismo por execução:** cada execução é independente e a seed depende só
  do cenário e da replicação (`Cenario.seed`), não da ordem nem do processo; então
  o resultado é o mesmo com 1 ou com vários processos. As execuções mais pesadas
  (N grande) são sorteadas primeiro, para os processos terminarem juntos.
- **Retomável:** uma execução é pulada se o manifesto existe, tem a seed e o
  cenário esperados (o campo `n_replicacoes` é ignorado na comparação, então
  aumentar de 5 para 10 replicações mantém as 5 primeiras) e o arquivo de dados
  confere (existência, linhas, SHA-256 e resumo; `manifesto.verificar_dados`).
  Manifesto ilegível, dado ausente ou adulterado: a execução é refeita. A
  conferência só lê arquivos, não simula de novo.
- **Interrupção segura:** o dado é gravado de forma atômica antes do manifesto
  (`saida.gravar_conteudo`); uma execução interrompida no meio não deixa arquivo
  incompleto com nome definitivo e é refeita na próxima chamada.
- **Falha isolada:** um erro em uma execução vai para a lista de falhas e as
  demais continuam; o código de saída é 1 se houve falha.
- **Árvore do git limpa:** o manifesto grava o commit usado; com alterações locais
  ele não reproduz o código. Por isso o script **recusa rodar** com a árvore suja
  (e avisa se não houver git), a menos que `--permitir-arvore-suja` seja dado
  (só para testes).
- **Progresso:** uma linha por execução concluída, com contagem, tempo da
  execução, tempo decorrido e estimativa de término.

Como usar (venv ativado, a partir da raiz TCC1, com `git status` limpo):

    python -m src.simulation.executar_grade --listar            # o que há, o que falta
    python -m src.simulation.executar_grade --processos 4       # roda o que falta
    python -m src.simulation.executar_grade --verificar         # confere o que existe
    python -m src.simulation.executar_grade --cenario N010_C06_xi0.50 --replicacao 0
    python -m src.simulation.teste_executar_grade

Depois da execução: `python -m src.data.marcacao --tudo` (marcação por limiar).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from src.simulation.cenarios import CAMINHO_PADRAO, Cenario, carregar_grade
from src.simulation.manifesto import (
    VERSAO_MANIFESTO,
    _info_git,
    carregar_manifesto,
    executar_e_registrar,
    nome_manifesto,
    verificar_dados,
)

RAIZ = Path(__file__).resolve().parents[2]
PASTA_PADRAO = RAIZ / "data" / "raw"


def planejar(
    cenarios,
    ids: list[str] | None = None,
    replicacoes: list[int] | None = None,
) -> list[tuple[Cenario, int]]:
    """Lista (cenário, replicação) a considerar, das mais pesadas para as mais leves.

    `ids` e `replicacoes` filtram; id desconhecido ou replicação fora do intervalo
    de algum cenário filtrado falha com ValueError.
    """
    por_id = {c.id: c for c in cenarios}
    if ids:
        desconhecidos = [i for i in ids if i not in por_id]
        if desconhecidos:
            raise ValueError(f"cenários desconhecidos: {desconhecidos}")
        cenarios = [por_id[i] for i in ids]
    tarefas = []
    for c in cenarios:
        reps = replicacoes if replicacoes else range(c.n_replicacoes)
        for k in reps:
            if not 0 <= k < c.n_replicacoes:
                raise ValueError(
                    f"{c.id}: replicação {k} fora do intervalo 0 a {c.n_replicacoes - 1}"
                )
            tarefas.append((c, k))
    # custo aproximado: cresce com N (chamadas por requisição) e com C (fila e laços)
    tarefas.sort(key=lambda t: (-t[0].n_fan_out, -t[0].concorrencia, t[0].xi, t[1]))
    return tarefas


def _sem_replicacoes(cenario: dict) -> dict:
    return {k: v for k, v in cenario.items() if k != "n_replicacoes"}


def execucao_conferida(cenario: Cenario, replicacao: int, pasta: Path | str) -> bool:
    """True se a execução já existe em `pasta` e confere (ver a docstring do módulo)."""
    pasta = Path(pasta)
    caminho = pasta / nome_manifesto(cenario, replicacao)
    if not caminho.exists():
        return False
    try:
        m = carregar_manifesto(caminho)
        if m.get("versao_manifesto") != VERSAO_MANIFESTO or m.get("dados") is None:
            return False
        if m["id"] != cenario.id or m["replicacao"] != replicacao:
            return False
        if m["seed"] != cenario.seed(replicacao):
            return False
        if _sem_replicacoes(m["cenario"]) != _sem_replicacoes(asdict(cenario)):
            return False
        return verificar_dados(m, pasta) == []
    except (OSError, ValueError, KeyError, TypeError):  # JSON ilegível, campo faltando, gzip
        return False


def classificar(tarefas, pasta: Path | str) -> tuple[list, list]:
    """Separa as tarefas em (conferidas, pendentes)."""
    conferidas, pendentes = [], []
    for c, k in tarefas:
        (conferidas if execucao_conferida(c, k, pasta) else pendentes).append((c, k))
    return conferidas, pendentes


def checar_git(info: dict | None = None, permitir_suja: bool = False) -> tuple[bool, str | None]:
    """(pode_rodar, mensagem). Recusa árvore suja; avisa se não houver git."""
    info = info if info is not None else _info_git()
    if info["commit"] is None:
        return True, "aviso: sem git; o manifesto não registrará o commit"
    if info["alteracoes_locais"]:
        if permitir_suja:
            return True, "aviso: árvore do git com alterações; o commit gravado não reproduz o código"
        return False, (
            "a árvore do git tem alterações locais: o commit gravado no manifesto não "
            "reproduziria o código. Faça o commit (e o push) antes, ou use "
            "--permitir-arvore-suja só para testes."
        )
    return True, None


def _executar_tarefa(cenario: Cenario, replicacao: int, pasta: str, grade: str) -> float:
    """Roda uma execução e devolve o tempo (s), incluindo a gravação."""
    t0 = time.perf_counter()
    executar_e_registrar(cenario, replicacao, pasta, grade)
    return time.perf_counter() - t0


def _formatar(segundos: float) -> str:
    s = int(round(segundos))
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}min" if h else f"{m}min{s:02d}s" if m else f"{s}s"


def executar(
    tarefas,
    pasta: Path | str = PASTA_PADRAO,
    grade: Path | str = CAMINHO_PADRAO,
    processos: int = 1,
    progresso=print,
) -> dict:
    """Roda as tarefas pendentes (já filtradas) e devolve o resumo.

    Devolve {'executadas': n, 'puladas': n, 'falhas': [(id, replicacao, motivo)], 'duracao_s': x}.
    """
    pasta, grade = Path(pasta), Path(grade)
    conferidas, pendentes = classificar(tarefas, pasta)
    total = len(pendentes)
    resumo = {"executadas": 0, "puladas": len(conferidas), "falhas": [], "duracao_s": 0.0}
    if progresso:
        progresso(
            f"{len(tarefas)} execuções consideradas: {len(conferidas)} já conferidas, "
            f"{total} a rodar com {processos} processo(s)"
        )
    if not pendentes:
        return resumo

    inicio = time.perf_counter()
    feitas = 0

    def registrar(c: Cenario, k: int, duracao: float | None, erro: Exception | None) -> None:
        nonlocal feitas
        feitas += 1
        if erro is not None:
            resumo["falhas"].append((c.id, k, f"{type(erro).__name__}: {erro}"))
            linha = f"FALHA {c.id} rep{k}: {type(erro).__name__}: {erro}"
        else:
            resumo["executadas"] += 1
            linha = f"{c.id} rep{k} em {duracao:.1f} s"
        if progresso:
            decorrido = time.perf_counter() - inicio
            falta = decorrido / feitas * (total - feitas)
            progresso(
                f"[{feitas}/{total}] {linha} | decorrido {_formatar(decorrido)}, "
                f"faltam cerca de {_formatar(falta)}"
            )

    if processos <= 1:
        for c, k in pendentes:
            try:
                registrar(c, k, _executar_tarefa(c, k, str(pasta), str(grade)), None)
            except Exception as erro:  # uma execução com problema não derruba as outras
                registrar(c, k, None, erro)
    else:
        with ProcessPoolExecutor(processos) as pool:
            futuros = {
                pool.submit(_executar_tarefa, c, k, str(pasta), str(grade)): (c, k)
                for c, k in pendentes
            }
            for futuro in as_completed(futuros):
                c, k = futuros[futuro]
                try:
                    registrar(c, k, futuro.result(), None)
                except Exception as erro:
                    registrar(c, k, None, erro)
    resumo["duracao_s"] = time.perf_counter() - inicio
    return resumo


def verificar_existentes(tarefas, pasta: Path | str = PASTA_PADRAO) -> dict:
    """Confere o que já existe, sem simular: {'conferidas', 'ausentes', 'problemas'}.

    Ausente = sem manifesto; problema = manifesto presente mas execução que não confere
    (dado ausente, adulterado, seed ou cenário diferente).
    """
    pasta = Path(pasta)
    conferidas, ausentes, problemas = 0, [], []
    for c, k in tarefas:
        if execucao_conferida(c, k, pasta):
            conferidas += 1
        elif not (pasta / nome_manifesto(c, k)).exists():
            ausentes.append((c.id, k))
        else:
            problemas.append((c.id, k))
    return {"conferidas": conferidas, "ausentes": ausentes, "problemas": problemas}


def processos_padrao() -> int:
    """Todos os núcleos menos um (deixa o computador utilizável), no mínimo 1."""
    return max(1, (os.cpu_count() or 2) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Execução da grade de cenários do TCC")
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--listar", action="store_true", help="mostra o que há e o que falta, sem rodar")
    modo.add_argument("--verificar", action="store_true", help="confere o que existe, sem rodar")
    ap.add_argument("--grade", default=str(CAMINHO_PADRAO), help="caminho do TOML")
    ap.add_argument("--pasta", default=str(PASTA_PADRAO), help="pasta de saída (data/raw)")
    ap.add_argument(
        "--processos", type=int, default=processos_padrao(),
        help=f"processos em paralelo (padrão: núcleos - 1 = {processos_padrao()})",
    )
    ap.add_argument("--cenario", action="append", metavar="ID", help="só este cenário (repetível)")
    ap.add_argument("--replicacao", action="append", type=int, metavar="K", help="só esta replicação (repetível)")
    ap.add_argument("--permitir-arvore-suja", action="store_true", help="só para testes")
    args = ap.parse_args()

    cenarios = carregar_grade(args.grade)
    tarefas = planejar(cenarios, args.cenario, args.replicacao)
    c0 = cenarios[0]

    if args.listar or args.verificar:
        r = verificar_existentes(tarefas, args.pasta)
        print(
            f"{len(cenarios)} cenários x {c0.n_replicacoes} replicações; consideradas "
            f"{len(tarefas)}: {r['conferidas']} conferidas, {len(r['ausentes'])} ausentes, "
            f"{len(r['problemas'])} com problema"
        )
        for id_, k in r["problemas"]:
            print(f"PROBLEMA {id_} rep{k}: manifesto presente, mas a execução não confere")
        if args.verificar:
            raise SystemExit(1 if r["problemas"] else 0)
        return

    pode, mensagem = checar_git(permitir_suja=args.permitir_arvore_suja)
    if mensagem:
        print(mensagem)
    if not pode:
        raise SystemExit(2)
    r = executar(tarefas, args.pasta, args.grade, args.processos)
    print(
        f"{r['executadas']} executadas, {r['puladas']} já conferidas, "
        f"{len(r['falhas'])} falhas, em {_formatar(r['duracao_s'])}"
    )
    for id_, k, motivo in r["falhas"]:
        print(f"FALHA {id_} rep{k}: {motivo}")
    if not r["falhas"]:
        print("Próximo passo: python -m src.data.marcacao --tudo")
    raise SystemExit(1 if r["falhas"] else 0)


if __name__ == "__main__":
    main()
