"""Verificação mínima do ambiente de simulação — Etapa 5, passo 1.

Este script NÃO é o modelo final. Ele só confirma que o SimPy está
instalado e funcionando corretamente no ambiente local (.venv), antes de
começar a modelar os recursos (passo 2), a geração de latência
lognormal+GPD (passo 3) e os processos de fan-out (passo 4).

Como rodar (com o venv ativado, a partir da raiz do projeto TCC1):

    python -m src.simulation.sandbox_check

Se aparecer a mensagem final "Ambiente SimPy funcionando corretamente.",
o ambiente está pronto para os próximos passos.
"""

import simpy


def chamada_de_exemplo(env: simpy.Environment, nome: str):
    """Processo mínimo: simula uma única "chamada" com duração fixa.

    Isso é só um teste de fumaça (smoke test) do SimPy — a duração fixa
    de 1.0 será substituída, no passo 3, por um sorteio da distribuição
    spliced (corpo lognormal + cauda GPD).
    """
    print(f"[{env.now:5.1f}] {nome}: chamada iniciada")
    yield env.timeout(1.0)
    print(f"[{env.now:5.1f}] {nome}: chamada concluída")


def main():
    env = simpy.Environment()
    env.process(chamada_de_exemplo(env, "api-exemplo"))
    env.run()
    print("\nAmbiente SimPy funcionando corretamente.")


if __name__ == "__main__":
    main()
