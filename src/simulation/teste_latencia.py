"""Verificação do gerador de latência — Etapa 5, passo 3.

Como rodar (venv ativado, a partir da raiz do projeto TCC1):

    python -m src.simulation.teste_latencia

Se aparecer "Latência funcionando corretamente.", o gerador está verificado.

Critério estatístico (definido antes de rodar): a amostra é fixa (seed
conhecida), então o resultado é determinístico. Para cada nível de xi e cada
p em {0,5; 0,9; 0,95; 0,99}, a fração da amostra abaixo do quantil teórico
deve estar a no máximo 3,29 desvios-padrão binomiais de p (intervalo de 99,9%).
Além disso, o teste de Kolmogorov-Smirnov contra a cdf teórica deve ter
p-valor > 0,001. Se falhar com outra seed, investigar o código; não afrouxar
a tolerância.
"""

import numpy as np
from scipy.stats import kstest

from src.simulation.latencia import ParamsLatencia

XIS = (0.2, 0.5, 0.8, 1.1)
PS = (0.5, 0.9, 0.95, 0.99)
N = 200_000
SEED = 2026
Z_999 = 3.29  # limite de 99,9% (bilateral) da normal padrão


def teste_cdf_e_quantil_sao_inversas():
    ps = np.array([0.001, 0.1, 0.5, 0.9, 0.95, 0.96, 0.99, 0.999, 0.9999])
    for xi in XIS:
        p = ParamsLatencia(xi=xi)
        assert np.allclose(p.cdf(p.quantil(ps)), ps, atol=1e-9), xi


def teste_cdf_continua_em_u():
    for xi in XIS:
        p = ParamsLatencia(xi=xi)
        assert abs(p.cdf(p.u - 1e-9) - p.p_corte) < 1e-8, xi
        assert abs(p.cdf(p.u + 1e-9) - p.p_corte) < 1e-8, xi


def teste_densidade_continua_em_u():
    """Efeito da escolha de beta por continuidade (decisão do TCC)."""
    for xi in XIS:
        p = ParamsLatencia(xi=xi)
        esq, dir_ = p.pdf(p.u - 1e-9), p.pdf(p.u + 1e-9)
        assert abs(esq - dir_) / esq < 1e-6, (xi, esq, dir_)


def teste_densidade_e_derivada_da_cdf():
    for xi in XIS:
        p = ParamsLatencia(xi=xi)
        for x in (5.0, 15.0, 30.0, 60.0, 200.0):  # corpo e cauda
            h = 1e-6 * x
            derivada = (p.cdf(x + h) - p.cdf(x - h)) / (2 * h)
            assert abs(derivada - p.pdf(x)) / p.pdf(x) < 1e-4, (xi, x)


def teste_mediana_e_corpo_sao_da_lognormal():
    p = ParamsLatencia()
    assert abs(p.quantil(0.5) - np.exp(p.mu)) < 1e-9
    assert abs(p.quantil(0.0)) == 0.0


def teste_p90_e_p95_fixos_e_p99_cresce_com_xi():
    """Propriedade da seção 2.2: só o p99 depende do peso da cauda."""
    p90 = {ParamsLatencia(xi=xi).quantil(0.90) for xi in XIS}
    p95 = {ParamsLatencia(xi=xi).quantil(0.95) for xi in XIS}
    assert len(p90) == 1 and len(p95) == 1, (p90, p95)
    p99 = [ParamsLatencia(xi=xi).quantil(0.99) for xi in XIS]
    assert p99 == sorted(p99) and len(set(p99)) == len(XIS), p99


def teste_amostra_bate_com_a_teoria():
    for xi in XIS:
        p = ParamsLatencia(xi=xi)
        x = p.sortear(np.random.default_rng(SEED), N)
        assert x.shape == (N,) and np.all(x >= 0)
        for prob in PS:
            frac = float(np.mean(x <= p.quantil(prob)))
            dp = np.sqrt(prob * (1 - prob) / N)
            assert abs(frac - prob) <= Z_999 * dp, (xi, prob, frac)
        frac_cauda = float(np.mean(x > p.u))
        dp = np.sqrt(0.05 * 0.95 / N)
        assert abs(frac_cauda - 0.05) <= Z_999 * dp, (xi, frac_cauda)
        assert kstest(x, p.cdf).pvalue > 0.001, xi


def teste_reprodutibilidade_com_a_mesma_seed():
    p = ParamsLatencia()
    a = p.sortear(np.random.default_rng(7), 1000)
    b = p.sortear(np.random.default_rng(7), 1000)
    c = p.sortear(np.random.default_rng(8), 1000)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def teste_formatos_de_saida():
    p = ParamsLatencia()
    rng = np.random.default_rng(1)
    assert isinstance(p.sortear(rng), float)
    assert p.sortear(rng, (2, 3)).shape == (2, 3)
    assert isinstance(p.quantil(0.99), float) and isinstance(p.cdf(30.0), float)


def teste_validacao_de_argumentos():
    for kwargs in ({"sigma": 0.0}, {"p_corte": 1.0}, {"p_corte": 0.0}, {"xi": 0.0}):
        try:
            ParamsLatencia(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"deveria falhar com {kwargs}")
    for p in (-0.1, 1.5):
        try:
            ParamsLatencia().quantil(p)
        except ValueError:
            continue
        raise AssertionError(f"quantil({p}) deveria falhar")


def main():
    testes = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"ok  {t.__name__}")
    print("\nLatência funcionando corretamente.")


if __name__ == "__main__":
    main()
