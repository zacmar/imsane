import torch as th
from time import perf_counter
import matplotlib.pyplot as plt
import math
import linop
import optim
import skimage.data as sd


def anisotv_l2():
    x = th.from_numpy(sd.shepp_logan_phantom())[None, None]
    sigma = 0.1
    y = x + sigma * th.randn_like(x)

    K = linop.Grad2d()

    lamda = 5e0
    max_iter = 5_000
    L_D = math.sqrt(8)
    tau = 0.01
    sigma = 1 / L_D**2 / tau

    def prox_g(x: th.Tensor, t: float) -> th.Tensor:
        return (x + t * lamda * y) / (1 + t * lamda)

    def prox_twoinfty(p: th.Tensor, _: float) -> th.Tensor:
        np = (p**2).sum((1,), keepdim=True).sqrt()
        return p / th.maximum(np, th.ones_like(np))

    def nabla(p):
        return K @ (K.T @ p / lamda - y)

    tau_fista = 1 / (L_D**2 / lamda)
    pstar = optim.fista(
        K @ y,
        nabla,
        lambda p: prox_twoinfty(p, tau),
        tau=tau_fista,
        max_iter=10_000,
    )
    xstar = y - K.T @ pstar / lamda
    i = 0

    def energy(x_):
        return ((x_ - y) ** 2).sum() / 2 * lamda + (K @ x_).abs().sum()

    fxstar = energy(xstar)

    def callback(x_, a):
        nonlocal i
        a[i] = energy(x_) - fxstar
        i += 1

    i = 0
    pdhg1_error = th.empty((max_iter,))
    t = perf_counter()
    _ = optim.pdhg1(
        y,
        K,
        tau,
        lambda x: prox_g(x, tau),
        sigma,
        lambda y: prox_twoinfty(y, sigma),
        max_iter=max_iter,
        callback=lambda x, _: callback(x, pdhg1_error),
    )
    print(f"PDHG1: {perf_counter() - t:.2f}")

    gamma = 0.35 * lamda
    i = 0
    pdhg2_error = th.empty((max_iter,))
    t = perf_counter()
    _ = optim.pdhg2(
        y,
        K,
        tau,
        prox_g,
        sigma,
        prox_twoinfty,
        gamma,
        max_iter=max_iter,
        callback=lambda x, _: callback(x, pdhg2_error),
    )
    print(f"PDHG2: {perf_counter() - t:.2f}")

    i = 0
    fista_error = th.empty((max_iter,))
    t = perf_counter()
    _ = optim.fista(
        K @ y,
        nabla,
        lambda p: prox_twoinfty(p, tau),
        tau=tau_fista,
        callback=lambda x: callback(y - K.T @ x / lamda, fista_error),
        max_iter=max_iter,
    )
    print(f"FISTA: {perf_counter() - t:.2f}")
    print('(unfair due to application of K.T in the callback)')
    plt.figure()
    plt.semilogy(pdhg1_error)
    plt.semilogy(pdhg2_error)
    plt.semilogy(fista_error)
    plt.legend(["pdgh1", "pdgh2", "fista"])
    plt.title("$k \\mapsto F(x_k) - F(x^*)$")
    plt.show()


if __name__ == "__main__":
    anisotv_l2()
