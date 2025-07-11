from typing import Callable
import math
import torch as th
from imsane.linop import LinOp


# TODO adapt all algos to the convention that the callback also encodes the termination condition
def no_op(*_):
    return False


def gd(
    x0: th.Tensor,
    grad: Callable[[th.Tensor], th.Tensor],
    tau: float,
    max_iter: int = 100,
    callback: Callable[[th.Tensor], bool | None] = no_op,
):
    x = x0.clone()
    for _ in range(max_iter):
        x -= tau * grad(x)
        callback(x)
    return x


def pdhg1(
    x0: th.Tensor,
    K: LinOp,
    tau: float,
    prox_tG: Callable[[th.Tensor], th.Tensor],
    sigma: float,
    prox_sF: Callable[[th.Tensor], th.Tensor],
    theta: float = 1.0,
    max_iter: int = 10_000,
    callback: Callable[[th.Tensor, th.Tensor], bool | None] = no_op,
) -> th.Tensor:
    """
    Algorithm 1 in https://doi.org/10.1007/s10851-010-0251-1
    """
    x = x0.clone()
    x_bar = x0.clone()
    y = K @ x0

    for _ in range(max_iter):
        x_prev = x.clone()
        y = prox_sF(y + sigma * K @ x_bar)
        x = prox_tG(x - tau * K.T @ y)
        x_bar = x + theta * (x - x_prev)
        if callback(x, y):
            break

    return x


def pdhg2(
    x0: th.Tensor,
    K: LinOp,
    tau0: float,
    prox_g: Callable[[th.Tensor, float], th.Tensor],
    sigma0: float,
    prox_f: Callable[[th.Tensor, float], th.Tensor],
    gamma: float,
    max_iter: int = 10_000,
    callback: Callable[[th.Tensor, th.Tensor], None] = no_op,
) -> th.Tensor:
    """
    Algorithm 2 in https://doi.org/10.1007/s10851-010-0251-1
    """
    x = x0.clone()
    x_bar = x0.clone()
    y = K @ x
    sigma, tau = sigma0, tau0
    for _ in range(max_iter):
        x_prev = x.clone()
        y = prox_f(y + sigma * K @ x_bar, sigma)
        x = prox_g(x - tau * K.T @ y, tau)
        theta = 1 / math.sqrt(1 + 2 * gamma * tau)
        tau = theta * tau
        sigma = sigma / theta
        x_bar = x + theta * (x - x_prev)
        callback(x, y)

    return x


def fista(
    x0: th.Tensor,
    nabla_f: Callable[[th.Tensor], th.Tensor],
    prox_g: Callable[[th.Tensor], th.Tensor],
    tau: float,
    max_iter: int = 10_000,
    callback: Callable[[th.Tensor], bool | None] = no_op,
) -> th.Tensor:
    x = x0.clone()
    y = x0.clone()
    x_prev = x0.clone()
    t = 1

    for _ in range(max_iter):
        x = prox_g(y - tau * nabla_f(y))
        t_new = (1 + math.sqrt(1 + 4 * t**2)) / 2
        y = x + (t - 1) / t_new * (x - x_prev)
        t = t_new
        x_prev = x.clone()
        if callback(x):
            break

    return x


def ipalm(
    x0s: tuple[th.Tensor, ...],
    energy: Callable[[*tuple[th.Tensor, ...]], th.Tensor],
    gradients: tuple[Callable[[*tuple[th.Tensor, ...]], th.Tensor], ...],
    proxs: tuple[Callable[[th.Tensor, th.Tensor], th.Tensor], ...],
    max_iter: int = 50,
    dims: tuple[int, ...] = (1, 2, 3),
    callback: Callable[[list[th.Tensor]], None] = no_op,
) -> list[th.Tensor]:
    n_variables = len(x0s)
    xs = [x.clone() for x in x0s]
    xs_old = [x.clone() for x in x0s]
    Ls = [xs[0].new_ones(xs[0].shape[0], 1, 1, 1) for _ in range(n_variables)]

    def ip(x: th.Tensor, y: th.Tensor):
        return th.sum(x * y, dim=dims, keepdim=True)

    for it in range(max_iter):
        beta = 1 / math.sqrt(2)  # it / (it + 3)
        # Gauss-Seidelized updates
        for i in range(n_variables):
            x_ = xs[i] + beta * (xs[i] - xs_old[i])
            # or = over-relaxed; this is not strictly necessary i think.
            # it just makes the impl much easier at the cost of some memory
            xs_or = [xs[i].clone() for i in range(n_variables)]
            xs_or[i] = x_
            value, gradient = energy(*xs_or), gradients[i](*xs_or)
            xs_old[i] = xs[i].clone()
            for _ in range(10):
                xs[i] = proxs[i](x_ - gradient / Ls[i], 1 / Ls[i])
                dx = xs[i] - x_
                bound = value + ip(dx, gradient) + Ls[i] / 2 * ip(dx, dx)
                if th.all((value_new := energy(*xs)) <= bound):
                    break
                Ls[i] = th.where(value_new <= bound, Ls[i], 2 * Ls[i])
            Ls[i] /= 1.5
        callback(xs)

        mean_relative_diff = 0
        for x, x_old in zip(xs, xs_old):
            rel_diff = (
                ((x - x_old) ** 2).sum((1, 2, 3)).sqrt()
                / (x_old**2).sum((1, 2, 3)).sqrt()
            ).mean()
            mean_relative_diff += rel_diff
        mean_relative_diff /= n_variables
        # print(mean_relative_diff)
        if mean_relative_diff < 5e-3:
            break

    return xs


def cg(
    x0: th.Tensor,
    A: LinOp,
    b: th.Tensor,
    tol: float = 1e-4,
    dims: tuple[int, ...] = (1, 2, 3),
    max_iter: int = 100,
    callback: Callable[[th.Tensor], bool | None] = no_op,
):
    x = x0.clone()
    r = b - A @ x
    p = r.clone()

    def ip(a, b):
        return th.sum(a.conj() * b, dim=dims, keepdim=True).real

    for _ in range(max_iter):
        Ap = A @ p
        alpha = ip(r, r) / ip(p, Ap)
        alpha[~th.isfinite(alpha)] = 0.0
        x += alpha * p
        r_prev = th.clone(r)
        r -= alpha * Ap
        beta = ip(r, r) / ip(r_prev, r_prev)
        beta[~th.isfinite(beta)] = 0.0
        p = r + beta * p
        if th.max((r**2).sum(dim=dims).sqrt()) < tol:
            break
        callback(x)

    return x


# TODO implement properly
def power_method(x0: th.Tensor, A: LinOp, max_iter: int = 100):
    x = x0.clone()

    for _ in range(max_iter):
        ax = A @ x
        x = ax / (ax.abs() ** 2).sum().sqrt()

    return x


def condat_vu(
    x_0: th.Tensor,
    y_0: th.Tensor,
    K: LinOp,
    prox_g: Callable[[th.Tensor], th.Tensor],
    prox_fs: Callable[[th.Tensor], th.Tensor],
    nabla_h: Callable[[th.Tensor], th.Tensor],
    tau: float,
    sigma: float,
    callback: Callable[[th.Tensor, th.Tensor], bool | None] = no_op,
    max_iter=100,
):
    x = x_0.clone()
    y = y_0.clone()

    for _ in range(max_iter):
        x_old = x.clone()
        x = prox_g(x - tau * (K.T @ y + nabla_h(x)))
        y = prox_fs(y + sigma * (K @ (2 * x - x_old)))
        callback(x, y)

    return x


def irgn(
    x0,
    jacobian,
    solve,
    max_iter: int = 100,
    callback=lambda _: None,
):
    x = x0.clone()
    dx = th.zeros_like(x)

    for _ in range(max_iter):
        # This basically doesnt actually do anything anymore since everything
        # is outsourced to "solve" (which might be a misnomer)
        dx = solve(jacobian(x), x)
        x += dx
        callback(x)

    return x


def pgn(x0, jacobian, solve, max_iter: int = 100, callback=lambda _: None):
    x = x0.clone()

    for _ in range(max_iter):
        # This basically doesnt actually do anything anymore since everything
        # is outsourced to "solve" (which might be a misnomer)
        x = solve(jacobian(x), x)
        callback(x)

    return x
