import numpy as np
import torch
from .hadamard import make_S_mat


class Solver(torch.nn.Module):
    def __init__(self, pixels, step_size=1.0, epsilon=1e-6, t_init=0.1, t_step=1.5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = torch.nn.Parameter(
            torch.tensor(epsilon, dtype=torch.float32),
            requires_grad=False,
        )
        self.step_size = torch.nn.Parameter(
            torch.tensor(step_size, dtype=torch.float32), 
            requires_grad=False,
        )
        self.pixels = torch.nn.Parameter(torch.stack([
                torch.tensor(image, dtype=torch.float32) for image in pixels
            ],
            axis=-1,
        )[...,None])
        self.n = torch.nn.Parameter(
            torch.tensor(self.pixels.shape[-2], dtype=torch.int32), 
            requires_grad=False
        )
        self.S = torch.nn.Parameter(
            torch.tensor(make_S_mat(self.n), dtype=torch.float32),
            requires_grad=False,
        )
        self.t = torch.nn.Parameter(
            torch.tensor(t_init, dtype=torch.float32),
            requires_grad=False,
        )
        self.t_step = torch.nn.Parameter(
            torch.tensor(t_step, dtype=torch.float32),
            requires_grad=False,
        )
        self.time_points = None
        self.initialize()

    def initialize(self):
        time_points_init = torch.linalg.inv(self.S) @ self.pixels
        time_points_init = torch.maximum(
            time_points_init,
            torch.ones_like(time_points_init),
        )
        self.time_points = torch.nn.Parameter(
            time_points_init,
            requires_grad=False,
        )

    @torch.no_grad()
    def increase_t(self):
        self.t = torch.nn.Parameter(
            torch.tensor(self.t * self.t_step, dtype=torch.float32),
            requires_grad=False,
        )

    @torch.no_grad()
    def set_log_barrier_weight(self, t):
        self.t = torch.nn.Parameter(
            torch.tensor(t, dtype=torch.float32),
            requires_grad=False,
        )

    @torch.no_grad()
    def newton_step(self):
        H = Solver.hessian(self.time_points, self.pixels, self.S, self.t)
        idx = torch.arange(self.n)
        Hinv = torch.linalg.inv(H)

        J = Solver.jacobian(self.time_points, self.pixels, self.S, self.t)
        step = -Hinv @ J
        time_points = self.time_points + step * self.step_size
        self.time_points = torch.nn.Parameter(
            time_points,
            requires_grad=False,
        )
        self.residual = torch.squeeze(J.swapaxes(-2, -1) @ Hinv @ J, (-1, -2))

    @staticmethod
    def hessian(x, k, S, t):
        return Solver.likelihood_hessian(x, k, S) + Solver.barrier_hessian(x, t)

    @staticmethod
    def jacobian(x, k, S, t):
        return Solver.likelihood_jacobian(x, k, S) + Solver.barrier_jacobian(x, t)

    @staticmethod
    def likelihood_hessian(x, k, S):
        rate = S@x
        inv_square_rate = torch.square(torch.reciprocal(rate))
        SS = S[...,:,None,:] * S[...,None,:,:]
        H = torch.einsum(
            "...a,...a,dea->...de", 
            k.squeeze(-1), 
            inv_square_rate.squeeze(-1),
            SS
        )
        return H

    @staticmethod
    def likelihood_jacobian(x, k, S):
        return -S.T @ (k / (S@x) -1)

    @staticmethod
    def barrier_jacobian(x, t):
        return -(1./t) * torch.reciprocal(x)

    @staticmethod
    def barrier_hessian(x, t):
        return torch.diag_embed((1./t) * torch.reciprocal(x*x).squeeze(-1))


