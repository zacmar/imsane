from abc import abstractmethod
from typing import Self, Callable, Literal
import torch as th


class LinOp:
    @abstractmethod
    def apply(self, x: th.Tensor) -> th.Tensor: ...

    @abstractmethod
    def applyT(self, y: th.Tensor) -> th.Tensor: ...

    @property
    def T(self):
        return Transpose(self)

    def __add__(self, other: Self):
        return Sum(self, other)

    def __radd__(self, other: Self):
        return Sum(self, other)

    def __sub__(self, other: Self):
        return Diff(self, other)

    def __rsub__(self, other: Self):
        return Diff(self, other)

    def __mul__(self, other: float):
        return ScalarMul(self, other)

    def __rmul__(self, other: float):
        return ScalarMul(self, other)

    def __matmul__[T: (LinOp, th.Tensor)](self, other: T) -> T:
        if isinstance(other, LinOp):
            return Composition(self, other)
        return self.apply(other)


class Composition(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2

    def apply(self, x: th.Tensor):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyT(self, y: th.Tensor):
        return self.LinOp2.applyT(self.LinOp1.applyT(y))


class Sum(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2

    def apply(self, x: th.Tensor) -> th.Tensor:
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyT(self, y: th.Tensor):
        return self.LinOp2.applyT(y) + self.LinOp1.applyT(y)


class Diff(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2

    def apply(self, x: th.Tensor) -> th.Tensor:
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return self.LinOp1.applyT(y) - self.LinOp2.applyT(y)


class ScalarMul(LinOp):
    def __init__(self, LinOp: LinOp, other: float):
        self.LinOp = LinOp
        self.scalar = other

    def apply(self, x: th.Tensor) -> th.Tensor:
        return self.LinOp.apply(x) * self.scalar

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return self.LinOp.applyT(y) * self.scalar


class Transpose(LinOp):
    def __init__(self, LinOpT: LinOp):
        self.LinOpT = LinOpT

    def apply(self, x: th.Tensor) -> th.Tensor:
        return self.LinOpT.applyT(x)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return self.LinOpT.apply(y)


# Actual operators
class Grad2d(LinOp):
    def apply(self, x: th.Tensor) -> th.Tensor:
        grad = th.zeros((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)
        grad[:, 0:1, :, :-1] += x[:, :, :, 1:] - x[:, :, :, :-1]
        grad[:, 1:2, :-1, :] += x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad

    def applyT(self, y: th.Tensor) -> th.Tensor:
        div = th.zeros((y.shape[0], 1, *y.shape[2:]), device=y.device, dtype=y.dtype)
        div[:, :, :, 1:] += y[:, 0, :, :-1]
        div[:, :, :, :-1] -= y[:, 0, :, :-1]
        div[:, :, 1:, :] += y[:, 1, :-1, :]
        div[:, :, :-1, :] -= y[:, 1, :-1, :]
        return div


class Sample(LinOp):
    def __init__(self, indices: th.Tensor):
        self.indices = indices

    def apply(self, x: th.Tensor) -> th.Tensor:
        return x[self.indices]

    def applyT(self, y: th.Tensor) -> th.Tensor:
        # TODO guess we need to take the dtype as input arg
        # TODO the zeros can be cached since theyre always at the same position
        rv = th.zeros_like(self.indices, dtype=th.complex128)
        rv[self.indices] = y
        return rv


class Id(LinOp):
    def apply(self, x: th.Tensor) -> th.Tensor:
        return x

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return y


class Fft(LinOp):
    def __init__(
        self,
        dim: tuple[int, ...] = (-2, -1),
        norm: Literal["forward", "backward", "ortho"] = "ortho",
    ):
        self.dim = dim
        self.norm = norm

    def apply(self, x: th.Tensor) -> th.Tensor:
        return th.fft.fftn(x, dim=self.dim, norm=self.norm)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return th.fft.ifftn(y, dim=self.dim, norm=self.norm)


class Ifft(LinOp):
    def __init__(
        self,
        dim: tuple[int, ...] = (-2, -1),
        norm: Literal["forward", "backward", "ortho"] = "ortho",
    ):
        self.dim = dim
        self.norm = norm

    def apply(self, x: th.Tensor) -> th.Tensor:
        return th.fft.ifftn(x, dim=self.dim, norm=self.norm)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return th.fft.fftn(y, dim=self.dim, norm=self.norm)


class Fftshift(LinOp):
    def __init__(self, dim: tuple[int, ...] = (-2, -1)):
        self.dim = dim

    def apply(self, x: th.Tensor) -> th.Tensor:
        return th.fft.fftshift(x, dim=self.dim)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return th.fft.ifftshift(y, dim=self.dim)


class Roll2(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.to(th.int64)

    def apply(self, x):
        n, _, h, w = x.shape
        c = self.shifts.shape[0]
        expanded = x.expand(-1, c, -1, -1)
        # TODO
        # https://discuss.pytorch.org/t/tensor-shifts-in-torch-roll/170655/2
        # This is still not really optimal, lots of stuff done for nothing
        # I think we can make this more efficient with a double-gather
        self.ind0 = th.arange(n, dtype=th.int64, device=x.device)[
            :, None, None, None
        ].expand(n, c, h, w)
        self.ind1 = th.arange(c, dtype=th.int64, device=x.device)[
            None, :, None, None
        ].expand(n, c, h, w)
        self.ind2 = th.arange(h, dtype=th.int64, device=x.device)[
            None, None, :, None
        ].expand(n, c, h, w)
        self.ind3 = th.arange(w, dtype=th.int64, device=x.device)[
            None, None, None, :
        ].expand(n, c, h, w)

        return expanded[
            self.ind0,
            self.ind1,
            (self.ind2 + self.shifts[None, :, 0, None, None]) % h,
            (self.ind3 + self.shifts[None, :, 1, None, None]) % w,
        ]

    def applyT(self, y):
        return y[
            self.ind0,
            self.ind1,
            (self.ind2 - self.shifts[None, :, 0, None, None]) % y.shape[2],
            (self.ind3 - self.shifts[None, :, 1, None, None]) % y.shape[3],
        ].sum(1, keepdim=True)


class Crop2(LinOp):
    def __init__(self, in_shape, crop_shape):
        self.in_shape = in_shape
        self.out_shape = crop_shape

        self.istart = (self.in_shape[0] - self.out_shape[0]) // 2
        self.iend = self.istart + self.out_shape[0]

        self.jstart = (self.in_shape[1] - self.out_shape[1]) // 2
        self.jend = self.jstart + self.out_shape[1]

        ipad2 = self.in_shape[0] - self.iend
        jpad2 = self.in_shape[1] - self.jend

        self.pads = tuple(int(pad) for pad in (self.jstart, jpad2, self.istart, ipad2))

    def apply(self, x):
        return x[:, :, self.istart : self.iend, self.jstart : self.jend]

    def applyT(self, y):
        return th.nn.functional.pad(y, self.pads, mode="constant")


class Ifftshift(LinOp):
    def __init__(self, dim: tuple[int, ...] = (-2, -1)):
        self.dim = dim

    def apply(self, x: th.Tensor) -> th.Tensor:
        return th.fft.ifftshift(x, dim=self.dim)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return th.fft.fftshift(y, dim=self.dim)


class Mul(LinOp):
    def __init__(self, coefs: th.Tensor):
        self.coefs = coefs

    def apply(self, x: th.Tensor) -> th.Tensor:
        return self.coefs * x

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return self.coefs.conj() * y


class SumReduce(LinOp):
    def __init__(self, dim, size):
        self.dim = dim
        self.size = size

    def apply(self, x):
        return x.sum(dim=self.dim, keepdim=True)

    # TODO this is not generic yet wrt dimension
    def applyT(self, y):
        return y.expand(y.shape[0], self.size, *y.shape[2:])


# TODO this is a hack. i guess this structure (with apply and applyT being abstract in the base class) is not optimal.
# how can we make it such that the user can just define their operator?
class Generic(LinOp):
    def __init__(
        self,
        apply: Callable[[th.Tensor], th.Tensor],
        applyT: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        self.apply = apply
        self.applyT = applyT


class Stack(LinOp):
    def __init__(self, linops):
        self.linops = linops
        # TODO
        self.in_shape = (0, 0)
        self.out_shape = (0, 0)

    def apply(self, x):
        return th.cat(tuple(linop.apply(x) for linop in self.linops), dim=1)

    # TODO this is not generic yet... probably the best implementation is just
    # with using lists.. (as in GLM?)
    def applyT(self, y):
        res = self.linops[0].applyT(y[:, 0:1, :, :])
        for idx, linop in enumerate(self.linops[1:], start=1):
            res += linop.applyT(y[:, idx : idx + 1, :, :])
        return res


class Real(LinOp):
    def apply(self, x):
        return x.real

    def applyT(self, y):
        return y + 0j


class ShiftInterp(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.fliplr()
        self.dtype = th.float32

    def apply(self, x):
        x_ = x.expand(self.shifts.shape[0], 1, *x.shape[2:])
        thetas = th.eye(2, 3, dtype=th.float32)[None].repeat(self.shifts.shape[0], 1, 1)
        thetas[:, :, 2] = self.shifts / x.shape[2] * 2.0
        grid = (
            th.nn.functional.affine_grid(
                thetas, (self.shifts.shape[0], 1, *x.shape[2:]), align_corners=False
            )
            .to(self.dtype)
            .to(x.device)
        )
        grid = ((grid + 1) % 2) - 1
        return (
            th.nn.functional.grid_sample(x_.real, grid, align_corners=False)
            + 1j * th.nn.functional.grid_sample(x_.imag, grid, align_corners=False)
        ).permute(1, 0, 2, 3)

    def applyT(self, y):
        y_ = y.permute(1, 0, 2, 3)
        thetas = th.eye(2, 3, dtype=th.float32)[None].repeat(self.shifts.shape[0], 1, 1)
        thetas[:, :, 2] = -self.shifts / y.shape[2] * 2
        grid = (
            th.nn.functional.affine_grid(
                thetas, (self.shifts.shape[0], 1, *y.shape[2:]), align_corners=False
            )
            .to(self.dtype)
            .to(y.device)
        )
        grid = ((grid + 1) % 2) - 1
        return (
            th.nn.functional.grid_sample(y_.real, grid, align_corners=False)
            + 1j * th.nn.functional.grid_sample(y_.imag, grid, align_corners=False)
        ).sum(0, keepdim=True)
