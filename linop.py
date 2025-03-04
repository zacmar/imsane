from abc import abstractmethod
from typing import Self
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
    def apply(self, x):
        grad = th.zeros((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)
        grad[:, 0:1, :, :-1] += x[:, :, :, 1:] - x[:, :, :, :-1]
        grad[:, 1:2, :-1, :] += x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad

    def applyT(self, y):
        div = th.zeros((y.shape[0], 1, *y.shape[2:]), device=y.device, dtype=y.dtype)
        div[:, :, :, 1:] += y[:, 0, :, :-1]
        div[:, :, :, :-1] -= y[:, 0, :, :-1]
        div[:, :, 1:, :] += y[:, 1, :-1, :]
        div[:, :, :-1, :] -= y[:, 1, :-1, :]
        return div
