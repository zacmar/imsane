import torch as th
from typing import Callable
import math


def ula(
    x0: th.Tensor,
    nabla_u: Callable[[th.Tensor], th.Tensor],
    tau: float,
    n_iter: int = 10_000,
    callback: Callable[[th.Tensor], None] = lambda _: None,
) -> th.Tensor:
    """
    Section 1.4.1. in https://projecteuclid.org/journals/bernoulli/volume-2/issue-4/Exponential-convergence-of-Langevin-distributions-and-their-discrete-approximations/bj/1178291835.full
    """
    x = x0.clone()
    for _ in range(n_iter):
        x = x - tau * nabla_u(x) + math.sqrt(2 * tau) * th.randn_like(x)
        callback(x)

    return x


def annealed_ula(
    x0: th.Tensor,
    nabla_u: Callable[[th.Tensor, th.Tensor], th.Tensor],
    taus: th.Tensor,
    max_iter: int = 10_000,
    callback: Callable[[th.Tensor], None] = lambda _: None,
) -> th.Tensor:
    """
    Algorithm 1 in https://papers.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf
    """
    x = x0.clone()
    for tau in taus:

        def nabla_su(x):
            return nabla_u(x, tau)

        x = ula(x, nabla_su, tau.item(), max_iter, callback)

    return x


def myula(
    x0: th.Tensor,
    nabla_f: Callable[[th.Tensor], th.Tensor],
    prox_g: Callable[[th.Tensor], th.Tensor],
    gamma: float,
    lamda: float,
    max_iter: int = 10_000,
    callback: Callable[[th.Tensor], None] = lambda _: None,
) -> th.Tensor:
    """
    Algorithm 1 in https://doi.org/10.1137/16M1108340
    """
    x = x0.clone()
    for _ in range(max_iter):
        x = (
            (1 - gamma / lamda) * x
            - gamma * nabla_f(x)
            + gamma / lamda * prox_g(x)
            + math.sqrt(2 * gamma) * th.randn_like(x)
        )
        callback(x)

    return x


def daz(
    x0: th.Tensor,
    nabla_f: Callable[[th.Tensor], th.Tensor],
    prox_g: Callable[[th.Tensor, th.Tensor], th.Tensor],
    ts: th.Tensor,
    taus: th.Tensor,
    K: int = 1_000,
    callback: Callable[[th.Tensor], None] = lambda _: None,
) -> th.Tensor:
    """
    Algorithm 1 in https://arxiv.org/pdf/2502.01358
    """
    x = x0.clone()
    for t_n, tau_n in zip(ts, taus):

        def prox_tg(x):
            return prox_g(x, t_n)

        x = myula(x, nabla_f, prox_tg, tau_n.item(), t_n.item(), K, callback)

    return x
