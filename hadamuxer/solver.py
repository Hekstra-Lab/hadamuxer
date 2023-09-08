import numpy as np
import torch
from .hadamard import make_S_mat


class Solver(torch.nn.Module):
    zero = 0.
    def __init__(
            self, 
            pixels, 
            timings=None, 
            step_size=0.5, 
            epsilon=1e-2, 
            t_init=1.0, 
            t_step=2.0, 
            **kwargs
        ):
        """
        Parameters
        ----------
        pixels : array
            An array of pixels with shape (num_images, num_x, num_y)
        timings : array
            A vector of timings for each image with shape (num_images,).
            To increase precision, scale the timings vector by 1/d where
            d=120 is a sensible choice. 
        step_size : float
            How big of a newton step to take with default 0.5
        epsilon : float
            Numerical precision with default 0.01.
        t_init : float
            Initial value for the log barrier weight with default 1.
        t_step : float
            Increment for the log barrier weight with default 2.
            t_{n+1} = t_step * t_{n}
        """
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
        S = make_S_mat(self.n)
        if timings is not None:
            S = S * timings
        self.S = torch.nn.Parameter(
            torch.tensor(S, dtype=torch.float32),
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
    def line_search(self, step):
        return step * self.step_size

    @torch.no_grad()
    def liXXne_search(self, step):
        # Never intentionally cross zero
        step_max = self.rate / step
        step_max = torch.where(step < 0., self.step_size, step_max)
        step_max = step_max.min(-2, keepdims=True)[0]
        step_max = torch.minimum(step_max, self.step_size)
        return step * step_max

    @torch.no_grad()
    def newton_step(self):
        H = Solver.hessian(self.rate, self.pixels, self.Sinv, self.t)
        Hinv = torch.linalg.inv(H)
        #Hinv = torch.linalg.inv((1. - epsilon) * H + epsilon * torch.eye(self.n, dtype=H.dtype, device=H.device)[...,:,:])

        J = Solver.jacobian(self.rate, self.pixels, self.Sinv, self.t)
        step = -Hinv @ J

        step = self.line_search(step)
        rate = self.rate + step

        self.rate = torch.nn.Parameter(
            rate,
            requires_grad=False,
        )
        residual = torch.squeeze(J.swapaxes(-2, -1) @ Hinv @ J, (-1, -2)) / 2.
        return residual,step

    @torch.no_grad()
    def solve_newton(self, max_iter=100):
        residual = np.inf
        for i in range(max_iter):
            residual,_ = self.newton_step()
            residual = residual.max()
            if residual < self.epsilon:
                break
        return residual

    @torch.no_grad()
    def solve(self, max_iter=1_000):
        residual = np.inf
        for i in range(max_iter):
            residual = self.solve_newton(max_iter=max_iter)
            rmax = residual.max()
            frac = (residual < self.epsilon).sum() / torch.numel(residual)
            print(f"t={self.t.detach().cpu().numpy()} ; max_residual={rmax.detach().cpu().numpy()} ; frac_in_tol={frac.detach().cpu().numpy()}")
            if (i+1) / self.t < self.epsilon:
                break
            self.increase_t()

    @staticmethod
    def hessian(rate, k, Sinv, t):
        H =  t * Solver.likelihood_hessian(rate, k) 
        H += Solver.barrier_hessian(rate, Sinv) 
        return H

    @staticmethod
    def jacobian(rate, k, Sinv, t):
        J =  t * Solver.likelihood_jacobian(rate, k) 
        J += Solver.barrier_jacobian(rate, Sinv)
        return J

    @staticmethod
    def likelihood_jacobian(rate, k):
        return -k * torch.reciprocal(rate + Solver.zero) + 1.

    @staticmethod
    def likelihood_hessian(rate, k):
        diagonal = k * torch.reciprocal(torch.square(rate) + Solver.zero)
        return torch.diag_embed(diagonal.squeeze(-1))

    @staticmethod
    def barrier_jacobian(rate, Sinv):
        J = -Sinv.T@(1 / (Sinv@rate + Solver.zero))
        J += -torch.reciprocal(rate + Solver.zero)
        return J

    @staticmethod
    def barrier_hessian(rate, Sinv):
        x = Sinv@rate + Solver.zero
        inv_square_x = torch.square(torch.reciprocal(x))
        SS = Sinv[...,:,None,:] * Sinv[...,None,:,:]
        H = torch.einsum(
            "...a,dea->...de", 
            inv_square_x.squeeze(-1),
            SS
        )

        H += torch.diag_embed(torch.reciprocal(rate * rate + Solver.zero).squeeze(-1))
        return H



