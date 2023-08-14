import numpy as np
import torch
from .hadamard import make_S_mat


class Solver(torch.nn.Module):
    def __init__(self, pixels, step_size=1.0, epsilon=1e-8, t_init=1.0, t_step=2., **kwargs):
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
        self.Sinv = torch.nn.Parameter(
            torch.linalg.inv(self.S),
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
        self.rate = torch.nn.Parameter(
            torch.ones_like(self.pixels),
            requires_grad=False,
        )

    @property
    def time_points(self):
        return self.Sinv @ self.rate

    @torch.no_grad()
    def increase_t(self):
        self.t = torch.nn.Parameter(
            torch.tensor((self.t * self.t_step).clone().detach(), dtype=torch.float32),
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
        H = Solver.hessian(self.rate, self.pixels, self.Sinv, self.t, self.epsilon)
        #Hinv = torch.linalg.inv(H)
        Hinv = torch.linalg.inv((1. - self.epsilon) * H + self.epsilon * torch.eye(self.n, dtype=H.dtype, device=H.device)[...,:,:])

        J = Solver.jacobian(self.rate, self.pixels, self.Sinv, self.t, self.epsilon)
        step = -Hinv @ J
        rate = self.rate + step * self.step_size
        self.rate = torch.nn.Parameter(
            rate,
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
            residual = self.solve_newton()
            rmax = residual.max()
            frac = (residual < self.epsilon).sum() / torch.numel(residual)
            print(f"t={self.t.detach().cpu().numpy()} ; max_residual={rmax.detach().cpu().numpy()} ; frac_in_tol={frac.detach().cpu().numpy()}")
            if (i+1) / self.t < self.epsilon:
                break
            self.increase_t()

    @staticmethod
    def hessian(rate, k, Sinv, t, epsilon):
        H =  t * Solver.likelihood_hessian(rate, k, epsilon) 
        H += Solver.barrier_hessian(rate, Sinv, epsilon) 
        return H

    @staticmethod
    def jacobian(rate, k, Sinv, t, epsilon):
        J =  t * Solver.likelihood_jacobian(rate, k, epsilon) 
        J += Solver.barrier_jacobian(rate, Sinv, epsilon)
        return J

    @staticmethod
    def likelihood_jacobian(rate, k, epsilon):
        return -k / (rate + epsilon) + 1.

    @staticmethod
    def likelihood_hessian(rate, k, epsilon):
        diagonal = k * torch.reciprocal(torch.square(rate) + epsilon)
        return torch.diag_embed(diagonal.squeeze(-1))

    @staticmethod
    def barrier_jacobian(rate, Sinv, epsilon):
        J = -Sinv.T@(1 / (Sinv@rate + epsilon))
        J += -torch.reciprocal(rate + epsilon)
        return J

    @staticmethod
    def barrier_hessian(rate, Sinv, epsilon):
        x = Sinv@rate
        inv_square_x = torch.square(torch.reciprocal(x + epsilon))
        SS = Sinv[...,:,None,:] * Sinv[...,None,:,:]
        H = torch.einsum(
            "...a,dea->...de", 
            inv_square_x.squeeze(-1),
            SS
        )

        H += torch.diag_embed(torch.reciprocal(rate * rate + epsilon).squeeze(-1))
        return H



