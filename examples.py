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

    def prox_infty(p: th.Tensor, _: float) -> th.Tensor:
        return p.clip(min=-1, max=1)

    def nabla(p):
        return K @ (K.T @ p / lamda - y)

    tau_fista = 1 / (L_D**2 / lamda)
    pstar = optim.fista(
        K @ y,
        nabla,
        lambda p: prox_infty(p, tau),
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
        lambda y: prox_infty(y, sigma),
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
        prox_infty,
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
        lambda p: prox_infty(p, tau),
        tau=tau_fista,
        callback=lambda x: callback(y - K.T @ x / lamda, fista_error),
        max_iter=max_iter,
    )
    print(f"FISTA: {perf_counter() - t:.2f}")
    print("(unfair due to application of K.T in the callback)")
    plt.figure()
    plt.semilogy(pdhg1_error)
    plt.semilogy(pdhg2_error)
    plt.semilogy(fista_error)
    plt.legend(["pdgh1", "pdgh2", "fista"])
    plt.title("$k \\mapsto F(x_k) - F(x^*)$")
    plt.show()


def mri():
    x = th.from_numpy(sd.shepp_logan_phantom())[None, None]
    sigma = 0.1
    F = linop.Generic(
        lambda x: th.fft.rfft2(x, norm="ortho"),
        lambda y: th.fft.irfft2(y, norm="ortho"),
    )
    fx = F @ x
    th.random.manual_seed(0)
    mask = th.rand(fx.shape) > 0.75
    M = linop.Sample(mask)
    A = M @ F
    ax = A @ x

    y = ax + sigma * th.randn_like(ax)
    K = linop.Grad2d()
    L_D = math.sqrt(8)
    lamda = 10
    max_iter = 100_000

    def energy(z):
        return ((A @ z - y).abs() ** 2).sum() / 2 * lamda + (K @ z).abs().sum()

    def prox_g(z: th.Tensor, t: float) -> th.Tensor:
        return F.T @ ((F @ z + t * lamda * M.T @ y) / (1 + t * lamda * mask))

    def prox_infty(p: th.Tensor, _: float) -> th.Tensor:
        return p.clip(min=-1, max=1)

    tau = 1 / L_D / 10
    sigma = 1 / L_D**2 / tau

    xstar = optim.pdhg1(
        A.T @ y,
        K,
        tau,
        lambda x: prox_g(x, tau),
        sigma,
        lambda y: prox_infty(y, sigma),
        max_iter=10_000,
        callback=lambda x, _: print(energy(x))
    )
    estar = energy(xstar)

    def cb(z, times, values):
        times.append(perf_counter() - t)
        values.append(energy(z) - estar)
        return times[-1] > 10.0

    times_pdhg, values_pdhg = [], []
    t = perf_counter()
    _ = optim.pdhg1(
        A.T @ y,
        K,
        tau,
        lambda x: prox_g(x, tau),
        sigma,
        lambda y: prox_infty(y, sigma),
        max_iter=max_iter,
        callback=lambda x, _: cb(x, times_pdhg, values_pdhg),
    )

    def nabla(z):
        return lamda * A.T @ (A @ z - y)

    pstar = K @ A.T @ y

    def prox(z):
        def nabla_inner(p):
            return K @ (K.T @ p / lamda - z)

        prec = 1 / (iters + 1) ** 2

        def pd_gap(q):
            x_ = z - K.T @ q / lamda
            primal = ((x_ - z) ** 2).sum() / 2 * lamda + (K @ x_).abs().sum()
            dual = (
                -((K.T @ q - lamda * z) ** 2).sum() / 2 / lamda
                + (z**2).sum() / 2 * lamda
            )
            return (primal - dual) < prec

        nonlocal pstar

        pstar = optim.fista(
            pstar,
            nabla_inner,
            lambda p: prox_infty(p, tau),
            tau=lamda / L_D**2,
            callback=pd_gap,
            max_iter=max_iter,
        )
        return z - K.T @ pstar / lamda

    iters = 0
    times_fista, values_fista = [], []
    t = perf_counter()
    _ = optim.fista(
        A.T @ y,
        nabla,
        prox,
        1 / lamda,
        callback=lambda x: cb(x, times_fista, values_fista),
    )
    plt.figure()
    plt.semilogy(times_pdhg, values_pdhg)
    plt.semilogy(times_fista, values_fista)
    plt.legend(["pdgh", "fista"])
    plt.title("$k \\mapsto F(x_k) - F(x^*)$")
    plt.show()


if __name__ == "__main__":
    mri()
