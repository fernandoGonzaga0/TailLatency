"""Geração de latência (corpo lognormal + cauda GPD) — Etapa 5, passo 3.

Distribuição "spliced" (emendada) de Kyung (2026, eq. 1): lognormal até o
ponto de corte u e GPD (Pareto generalizada) acima dele,

    F(x) = F_LN(x)                              para x <= u
    F(x) = p_c + (1 - p_c) * G(x - u)           para x >  u

com p_c = F_LN(u), a fração de chamadas que ficam no corpo, e

    G(y) = 1 - (1 + xi*y/beta)^(-1/xi),  y >= 0        (Kyung, eq. 5).

Parâmetros livres: mu e sigma (lognormal), p_corte (= p_c) e xi (peso da
cauda). O ponto de corte é u = quantil p_corte da lognormal.

Escolha do TCC (não é do Kyung)
-------------------------------
No Kyung o scale beta é estimado por máxima verossimilhança. Aqui ele é
calculado para que a densidade também seja contínua em u:

    beta = (1 - p_c) / f_LN(u)

Assim o peso da cauda é controlado só pelo xi, e o beta não depende dele.
Essa decisão (21/10/2026) deve constar em Materiais e Métodos como escolha
do trabalho.

Por que p_corte = 0.95
----------------------
Com o corpo (mu, sigma) e o corte fixos, o p90 e o p95 individuais ficam no
corpo e não mudam com o xi; só o p99 cai na GPD e varia com ele (seção 2.2 do
documento de contexto). Um corte abaixo de 0.95 colocaria o p95 na cauda.

Uso
---
    import numpy as np
    from src.simulation.latencia import ParamsLatencia

    p = ParamsLatencia(xi=0.5)
    rng = np.random.default_rng(42)   # o gerador é recebido de fora (passo 6)
    tempos = p.sortear(rng, 1000)     # tempos de serviço, em ms
    limiar_p95 = p.quantil(0.95)      # forma fechada, sem simular (Fase 5)

O tempo simulado é adimensional; por convenção, 1.0 = 1 ms. Os valores
padrão (mediana de 10 ms, sigma = 0.5) são ilustrativos: o limiar de SLA é
relativo à própria distribuição, então o valor absoluto não afeta as
conclusões.

Validação
---------
`teste_latencia.py` (testes automáticos: inversa cdf/quantil, continuidade da
cdf e da densidade, p90/p95 fixos e p99 crescente com o xi, e amostra contra a
teoria) e `validar_latencia.py` (histograma, QQ-plot e curva de
sobrevivência em escala log-log, salvos em results/). A integração com
`recursos.chamar_api` e a seed por cenário vêm nos passos 4 e 6.
"""

from dataclasses import dataclass
from math import log

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ParamsLatencia:
    """Parâmetros do gerador de latência de uma API (iguais para todas as
    APIs de um cenário)."""

    mu: float = log(10.0)    # média do log da latência (mediana = exp(mu) = 10 ms)
    sigma: float = 0.5       # desvio do log da latência (> 0)
    p_corte: float = 0.95    # p_c: fração no corpo; o corte é o quantil p_c
    xi: float = 0.5          # parâmetro de forma da GPD (> 0: cauda pesada)

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma deve ser > 0")
        if not 0.0 < self.p_corte < 1.0:
            raise ValueError("p_corte deve estar em (0, 1)")
        if self.xi <= 0:
            raise ValueError("xi deve ser > 0 (cauda pesada)")

    # --- grandezas derivadas -------------------------------------------

    @property
    def u(self) -> float:
        """Ponto de corte: quantil p_corte da lognormal (em ms)."""
        return float(np.exp(self.mu + self.sigma * norm.ppf(self.p_corte)))

    @property
    def beta(self) -> float:
        """Scale da GPD, por continuidade da densidade em u (ver cabeçalho)."""
        z = norm.ppf(self.p_corte)
        f_ln_u = norm.pdf(z) / (self.sigma * self.u)  # densidade da lognormal em u
        return float((1.0 - self.p_corte) / f_ln_u)

    # --- distribuição --------------------------------------------------

    def cdf(self, x):
        """Função de distribuição acumulada F(x) (eq. 1 de Kyung)."""
        x = np.asarray(x, dtype=float)
        u, beta = self.u, self.beta
        with np.errstate(divide="ignore"):
            corpo = norm.cdf((np.log(np.maximum(x, 0.0)) - self.mu) / self.sigma)
        y = np.maximum(x - u, 0.0)
        g = 1.0 - (1.0 + self.xi * y / beta) ** (-1.0 / self.xi)  # eq. 5
        r = np.where(x <= u, corpo, self.p_corte + (1.0 - self.p_corte) * g)
        return float(r) if r.ndim == 0 else r

    def pdf(self, x):
        """Densidade f(x): lognormal até u e, acima de u, (1 - p_c) * g(x - u).

        Com o beta por continuidade, f(u-) = f(u+) para qualquer xi.
        """
        x = np.asarray(x, dtype=float)
        u, beta = self.u, self.beta
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = np.maximum(x, np.finfo(float).tiny)
            z = (np.log(xs) - self.mu) / self.sigma
            corpo = norm.pdf(z) / (self.sigma * xs)
        y = np.maximum(x - u, 0.0)
        cauda = (1.0 - self.p_corte) / beta * (1.0 + self.xi * y / beta) ** (-1.0 / self.xi - 1.0)
        r = np.where(x <= 0.0, 0.0, np.where(x <= u, corpo, cauda))
        return float(r) if r.ndim == 0 else r

    def quantil(self, p):
        """Quantil em forma fechada (inversa da cdf), p em [0, 1].

        Serve ao sorteio (transformada inversa) e ao cálculo analítico dos
        limiares de SLA (p90, p95, p99) sem simular.
        """
        p = np.asarray(p, dtype=float)
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("p deve estar em [0, 1]")
        u, beta, pc = self.u, self.beta, self.p_corte
        with np.errstate(divide="ignore"):
            corpo = np.exp(self.mu + self.sigma * norm.ppf(np.minimum(p, pc)))
            v = np.clip((p - pc) / (1.0 - pc), 0.0, 1.0)
            cauda = u + beta * np.expm1(-self.xi * np.log1p(-v)) / self.xi
        r = np.where(p < pc, corpo, cauda)
        return float(r) if r.ndim == 0 else r

    def sortear(self, rng: np.random.Generator, tamanho=None):
        """Sorteia tempos de serviço (ms) por transformada inversa.

        rng: gerador do NumPy (np.random.default_rng(seed)), recebido de fora
        para que a seed fique sob controle do cenário (passo 6).
        tamanho: None devolve um float; um inteiro ou tupla devolve um array.
        """
        return self.quantil(rng.random(tamanho))
