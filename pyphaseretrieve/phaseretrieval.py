import torch as th
import optim
import numpy as np
import linop


class Microscope:
    def __init__(
        self,
        led_positions: th.Tensor,
        camera_size: int,
        lamda: float = 0.514,
        na: float = 0.19,
        magnification: float = 10,
        pixel_size: float = 6.5,
    ):
        # Experiment Setup Parameters Setting (distance and length unit: um)
        # Optical system
        self.lamda = lamda
        self.na = na
        # Camera system
        self.camera_size = camera_size
        self.magnification = magnification

        img_pixel_size = pixel_size / self.magnification
        fov = img_pixel_size * self.camera_size
        self.fourier_res = 1 / fov

        if 2 * self.na / self.lamda > (1 / img_pixel_size) / 2:
            print("There is aliasing")

        camera_size_idx = th.arange(
            -np.floor(self.camera_size / 2),
            np.ceil(self.camera_size / 2),
            device=led_positions.device,
        )
        temp_mask_h, temp_mask_v = th.meshgrid(
            camera_size_idx, camera_size_idx, indexing="ij"
        )
        pupil_radius = self.na / self.lamda / self.fourier_res
        self.pupil_mask = linop.Fftshift() @ (
            th.sqrt(temp_mask_h**2 + temp_mask_v**2) <= pupil_radius
        )

        self.led_positions = led_positions
        sin_theta = led_positions[:, 0:2] / th.sqrt(
            (led_positions**2).sum(-1, keepdim=True)
        )
        self.led_na = (sin_theta**2).sum(-1).sqrt()
        self.shifts = sin_theta / self.lamda / self.fourier_res

    # TODO check correctness, looks very weird; taken from:
    # self.bf_mask = th.sqrt((sin_theta**2).sum(-1)) <= self.na
    def is_brightfield_led(self, position: th.Tensor) -> th.Tensor:
        tmp = position[:2] / (position**2).sum().sqrt()
        angle = th.angle(tmp[0] + 1j * tmp[1])
        return angle <= self.na

    # TODO check correctness
    # TODO change name, something like `max_unaliased_grid`?
    def reconstruction_size(self, indices: th.Tensor) -> int:
        na_illu = self.led_na[indices].max()
        na = na_illu + self.na
        return np.maximum(
            int(self.camera_size), int(th.ceil(2 * na / self.lamda / self.fourier_res))
        )


class MultiplexedFourierPtychography:
    def __init__(
        self,
        microsope: Microscope,
        indices: list[th.Tensor],
        shape: tuple[int, int],
        exact_phase: bool = False,
    ):
        self.probe = microsope.pupil_mask[None, None]
        self.n_leds = [len(index) for index in indices]
        self.n_patterns = len(indices)
        self.shifts = [microsope.shifts[index] for index in indices]

        phase_shift_ffts = (
            [linop.ShiftInterp(sh) @ linop.Fft() for sh in self.shifts]
            if exact_phase
            else [linop.Roll2(th.round(sh).to(th.int64)) @ linop.Fft() for sh in self.shifts]
        )

        self.forwards = [
            linop.Ifft()
            @ linop.Mul(self.probe)
            @ linop.Ifftshift()
            @ linop.Crop2(in_shape=shape, crop_shape=self.probe.shape[2:])
            @ linop.Fftshift()
            @ ps_fft
            for ps_fft in phase_shift_ffts
        ]

    def forward(self, x: th.Tensor):
        y = th.empty(
            (x.shape[0], self.n_patterns, *self.probe.shape[2:]),
            device=x.device,
        )
        for i, forward_ in enumerate(self.forwards):
            y[:, i] = (th.abs(forward_ @ x) ** 2).sum(1)

        return y

    def jacobian(self, x: th.Tensor):
        return linop.Stack(
            [
                linop.SumReduce(1, n_leds)
                @ linop.Real()
                @ linop.Mul(2 * (forward @ x).conj())
                @ forward
                for forward, n_leds in zip(self.forwards, self.n_leds)
            ]
        )


def FPM(
    y: th.Tensor,
    model: MultiplexedFourierPtychography,
    shape: tuple[int, int],
    n_iter: int = 50,
    tau: float = 1e-6,
    loss: str = "amplitude",
    epsilon: float = 1e-4,
):
    assert loss in ["amplitude", "intensity"]
    dtype = th.complex64 if y.dtype is th.float32 else th.complex128
    x0 = th.ones((1, 1, *shape), dtype=dtype, device=y.device)
    # x0 *= (th.mean(y[0, 0]) / model.forward(x0)[0, 0].mean()).sqrt()

    def nabla(x: th.Tensor):
        if loss == "amplitude":
            # This was tested against autograd, seems to be correct
            field = th.cat(tuple(forw @ x for forw in model.forwards), dim=1)
            ex = field - field / (field.abs() + epsilon) * y.sqrt()
            back = th.cat(
                tuple(forw.T @ e[None, None] for forw, e in zip(model.forwards, ex[0])),
                dim=1,
            )
            return back.sum(1, keepdim=True)
        else:
            return model.jacobian(x).T @ (model.forward(x) - y)

    return optim.gd(x0, nabla, tau, n_iter)


def DPC(
    y: th.Tensor,
    model: MultiplexedFourierPtychography,
    shape: tuple[int, int],
    alpha: float = 5e1,
) -> th.Tensor:
    dtype = th.complex64 if y.dtype is th.float32 else th.complex128
    crop = linop.Crop2(
        in_shape=shape,
        crop_shape=model.probe.shape[2:],
    )
    fac = shape[0] / model.probe.shape[2]

    numerator = th.zeros(shape, dtype=dtype, device=y.device)[None, None]
    denom = th.zeros(shape, dtype=y.dtype, device=y.device)[None, None]
    probe_ = model.probe.to(th.int32)
    pad = (shape[0] - probe_.shape[2]) // 2
    probe_ = th.fft.fftshift(
        th.nn.functional.pad(th.fft.ifftshift(probe_[0, 0]), (pad, pad, pad, pad))
    )[None, None]
    all_shifts = [th.round(sh).to(th.int64) for sh in model.shifts]

    for i_m, shifts in enumerate(all_shifts):
        hm = (linop.Roll2(-shifts) @ probe_ - linop.Roll2(shifts) @ probe_).sum(
            1, keepdim=True
        )
        numerator += (
            (hm * 1j).conj()
            * linop.Ifftshift()
            @ crop.T
            @ linop.Fftshift()
            @ linop.Fft()
            @ y[:, i_m : i_m + 1]
        )
        denom += th.abs(hm) ** 2

    return (linop.Ifft() @ (numerator / (denom + alpha))).real / fac


def PPR(
    y: th.Tensor,
    model: MultiplexedFourierPtychography,
    shape: tuple[int, int],
    n_iter: int = 4,
    inner_iter: int = 100,
    alpha: float = 1e5,
    reg="l2",
) -> th.Tensor:
    dtype = th.complex64 if y.dtype is th.float32 else th.complex128
    x0 = th.ones((1, 1, *shape), dtype=dtype, device=y.device)
    x0 *= (th.mean(y[0, 0]) / model.forward(x0)[0, 0].mean()).sqrt()

    def solve(J, x):
        nonlocal alpha
        alpha /= 1.2
        if reg == "tv":
            b = optim.power_method(J.T @ J, x, max_iter=10)
            opnormJTJ = (b * ((J.T @ J) @ b)).real.sum() / (b * b).real.sum()
            opnormD = np.sqrt(8)
            sigma = 1 / opnormD
            fac = th.sqrt(opnormJTJ)
            sigma *= fac
            sigmaLsqlH = sigma * opnormD**2 + opnormJTJ
            tau = 1 / sigmaLsqlH

            def nabla_h(x_):
                return J.T @ (J @ x_ + model.forward(x) - y)

            def prox_g(x):
                return x

            def prox_fs(y):
                y_ = y + alpha * sigma * linop.Grad2d() @ x
                return y_ / th.maximum(
                    (y_.abs() ** 2).sum(1, keepdims=True).sqrt() / alpha,
                    th.ones(y.shape, device=y.device),
                )

            return optim.condat_vu(
                th.zeros_like(x),
                th.zeros_like(linop.Grad2d() @ x),
                linop.Grad2d(),
                prox_g,
                prox_fs,
                nabla_h,
                tau,
                sigma,
                max_iter=inner_iter,
            )
        else:
            if reg == 'none':
                alpha = 0
            return optim.cg(
                th.zeros_like(x),
                J.T @ J + alpha * linop.Id(),
                J.T @ (y - model.forward(x)) - alpha * x,
                max_iter=inner_iter,
            )

    return optim.irgn(x0, model.jacobian, solve=solve, max_iter=n_iter)


def PPR_PGD(
    y: th.Tensor,
    model: MultiplexedFourierPtychography,
    shape: tuple[int, int],
    n_iter: int = 4,
    inner_iter: int = 100,
    alpha: float = 1e5,
    reg="l2",
) -> th.Tensor:
    dtype = th.complex64 if y.dtype is th.float32 else th.complex128
    x0 = th.ones((1, 1, *shape), dtype=dtype, device=y.device)
    x0 *= (th.mean(y[0, 0]) / model.forward(x0)[0, 0].mean()).sqrt()

    def solve(J, x_k):
        nonlocal alpha
        if reg == "tv":
            b = optim.power_method(J.T @ J, x_k, max_iter=10)
            opnormJTJ = (b * ((J.T @ J) @ b)).real.sum() / (b * b).real.sum()
            opnormD = np.sqrt(8)
            sigma = th.sqrt(opnormJTJ) / opnormD
            sigmaLsqlH = sigma * opnormD**2 + opnormJTJ
            tau = 1 / sigmaLsqlH

            def nabla_h(x):
                return J.T @ (J @ (x - x_k) + model.forward(x_k) - y)

            def prox_g(x):
                return x

            def prox_fs(y):
                return y / th.maximum(
                    (y.abs() ** 2).sum(1, keepdims=True).sqrt() / alpha,
                    th.ones(y.shape, device=y.device),
                )

            return optim.condat_vu(
                x_k,
                linop.Grad2d() @ x_k,
                linop.Grad2d(),
                prox_g,
                prox_fs,
                nabla_h,
                tau,
                sigma,
                max_iter=inner_iter,
            )
        else:
            if reg == 'none':
                alpha = 0
            return optim.cg(
                x_k,
                J.T @ J + alpha * linop.Id(),
                J.T @ (J @ x_k + y - model.forward(x_k)),
                max_iter=inner_iter,
            )

    return optim.pgn(x0, model.jacobian, solve=solve, max_iter=n_iter)
