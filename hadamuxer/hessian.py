import numpy as np
import torch
from hadamard import make_S_mat

# First test on the smol data
n = 11

S = torch.tensor(make_S_mat(n).astype('float32'))
Sinv = torch.linalg.inv(S)
x_true = torch.tensor(np.random.random((n, 1)).astype('float32')) + 1.
rate_true = S @ x_true
rate = torch.tensor(torch.ones_like(x_true), requires_grad=True)
k = torch.poisson(rate_true)

###############################################################################
# loss function for maximum likelihood
###############################################################################

def loss_fn(rate):
    loss = (k * torch.log(rate) - rate).sum()
    loss = torch.squeeze(loss)
    return -loss

def custom_jacobian(rate):
    return -k / rate + 1.

def custom_hessian(rate):
    diagonal = k * torch.reciprocal(torch.square(rate))
    return torch.diag_embed(diagonal.squeeze(-1))

J = torch.autograd.functional.jacobian(loss_fn, (rate,))[0]
H = torch.autograd.functional.hessian(loss_fn, (rate,))[0][0]
H = torch.squeeze(H)

assert torch.allclose(J, custom_jacobian(rate))
assert torch.allclose(H, custom_hessian(rate))


###############################################################################
# Log barrier for the x=Sinv@rate
###############################################################################

# strength of the barrier function
t = np.random.random()

def log_barrier(rate):
    loss = -(1/t) * torch.log(Sinv@rate)
    return loss.sum()


J = torch.autograd.functional.jacobian(log_barrier, (rate,))[0]
H = torch.autograd.functional.hessian(log_barrier, (rate,))[0][0]
H = torch.squeeze(H)

def custom_jacobian(rate):
    return -(1/t) * Sinv.T@(1 / (Sinv@rate))

def custom_hessian(rate):
    x = Sinv@rate
    inv_square_x = torch.square(torch.reciprocal(x))
    SS = Sinv[...,:,None,:] * Sinv[...,None,:,:]
    H = torch.einsum(
        "...a,dea->...de", 
        inv_square_x.squeeze(-1),
        SS
    )
    return (1. / t) * H

assert torch.allclose(J, custom_jacobian(rate))
assert torch.allclose(H, custom_hessian(rate))

