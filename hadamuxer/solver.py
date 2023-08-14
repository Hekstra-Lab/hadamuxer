import numpy as np
import torch
from .hadamard import make_S_mat


class Solver(torch.nn.Module):
    def __init__(self, pixels, step_size=1.0, epsilon=1e-5, t_init=0.1, t_step=1.1, **kwargs):
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

    @property
    def rate(self):
        return self.S @ self.time_points

    def initialize(self):
        time_points_init = torch.ones_like(self.pixels)
        #time_points_init = torch.linalg.inv(self.S) @ self.pixels
        #time_points_init = torch.maximum(
        #    time_points_init,
        #    torch.ones_like(time_points_init),
        #)
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
        H = Solver.hessian(self.time_points, self.pixels, self.S, self.t, self.epsilon)
        Hinv = torch.linalg.inv(H)

        J = Solver.jacobian(self.time_points, self.pixels, self.S, self.t, self.epsilon)
        step = -Hinv @ J
        time_points = self.time_points + step * self.step_size
        self.time_points = torch.nn.Parameter(
            time_points,
            requires_grad=False,
        )
        residual = torch.squeeze(J.swapaxes(-2, -1) @ Hinv @ J, (-1, -2))
        return residual

    @torch.no_grad()
    def solve_newton(self, max_iter=100):
        self.epsilon
        residual = np.inf
        for i in range(max_iter):
            residual = self.newton_step().max()
            if residual < self.epsilon:
                break
        return residual

    @torch.no_grad()
    def solve(self, max_iter=1_000):
        residual = np.inf
        for i in range(max_iter):
            residual = self.solve_newton().max()
            print(f"t={self.t.detach().cpu().numpy()} ; residual={residual.detach().cpu().numpy()})")
            if (i+1) / self.t < self.epsilon:
                break
            self.increase_t()

    @staticmethod
    def hessian(x, k, S, t, epsilon):
        H = t * Solver.likelihood_hessian(x, k, S) 
        H += Solver.time_point_barrier_hessian(x) 
        H += Solver.rate_barrier_hessian(x, S)
        return H

    @staticmethod
    def jacobian(x, k, S, t, epsilon):
        J = t * Solver.likelihood_jacobian(x, k, S) 
        J += Solver.time_point_barrier_jacobian(x)
        J += Solver.rate_barrier_jacobian(x, S)
        return J

    @staticmethod
    def likelihood_jacobian(x, k, S):
        return -S.T @ (k / (S@x) -1)

    @staticmethod
    def likelihood_hessian(x, k, S):
        rate = S@x
        assert torch.all(rate > 0.)
        inv_square_rate = torch.reciprocal(torch.square(rate))
        SS = S[...,:,None,:] * S[...,None,:,:]
        H = torch.einsum(
            "...a,...a,dea->...de", 
            k.squeeze(-1), 
            inv_square_rate.squeeze(-1),
            SS
        )
        return H

    @staticmethod
    def time_point_barrier_jacobian(x):
        return -torch.reciprocal(x)

    @staticmethod
    def time_point_barrier_hessian(x):
        return torch.diag_embed(torch.reciprocal(x*x)).squeeze(-1)

    @staticmethod
    def rate_barrier_jacobian(x, S):
        """
        Ensures that the model predicts positive rate=S@x
        """
        return -S.T@(1 / (S@x))

    @staticmethod
    def rate_barrier_hessian(x, S):
        """
        Ensures that the model predicts positive rate=S@x
        """
        rate = S@x
        inv_square_rate = torch.reciprocal(torch.square(rate))
        SS = S[...,:,None,:] * S[...,None,:,:]
        H = torch.einsum(
            "...a,dea->...de", 
            inv_square_rate.squeeze(-1),
            SS
        )
        return H




