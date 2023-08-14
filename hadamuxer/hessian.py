import numpy as np
import torch
from hadamard import make_S_mat

# First test on the smol data
n = 11
x = torch.tensor(np.random.random((n, 1)).astype('float32'), requires_grad=True)
k = torch.tensor(np.random.random((n, 1)).astype('float32'))
S = torch.tensor(make_S_mat(n).astype('float32'))


def loss_fn(x):
    loss = (k * torch.log(S@x) - S@x).sum()
    loss = torch.squeeze(loss)
    return -loss

def custom_jacobian(x, k):
    return -S.T @ (k / (S@x) -1)

def custom_hessian(x, k):
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

J = torch.autograd.functional.jacobian(loss_fn, (x,))[0]
H = torch.autograd.functional.hessian(loss_fn, (x,))[0][0]
H = torch.squeeze(H)

assert torch.allclose(J, custom_jacobian(x, k))
assert torch.allclose(H, custom_hessian(x, k))

# Now try the big big
xdmin,ydmin = 2527, 2463
x = torch.tensor(np.random.random((xdmin, ydmin, n, 1)).astype('float32'))
k = torch.tensor(np.random.random((xdmin, ydmin, n, 1)).astype('float32'))

from time import time
start = time()
J = custom_jacobian(x, k)
H = custom_hessian(x, k)
stop = time()
print(f"{stop - start} s")
print(f"Hessian shape: {H.shape}")
print(f"Jacobian shape: {J.shape}")

# Back to smol
x = torch.tensor(np.random.random((n, 1)).astype('float32'), requires_grad=True)


# strength of the barrier function
t = np.random.random()

def log_barrier(x):
    loss = -(1/t) * torch.log(x)
    return loss.sum()


J = torch.autograd.functional.jacobian(log_barrier, (x,))[0]
H = torch.autograd.functional.hessian(log_barrier, (x,))[0][0]
H = torch.squeeze(H)

def custom_jacobian(x):
    return -(1./t) * torch.reciprocal(x)

def custom_hessian(x):
    return torch.diag_embed((1./t) * torch.reciprocal(x*x).squeeze(-1))


assert torch.allclose(J, custom_jacobian(x))
assert torch.allclose(H, custom_hessian(x))


# Now try the big big
xdmin,ydmin = 2527, 2463
x = torch.tensor(np.random.random((xdmin, ydmin, n, 1)).astype('float32'))
H = custom_hessian(x)

from time import time
start = time()
J = custom_jacobian(x)
H = custom_hessian(x)
stop = time()
print(f"{stop - start} s")
print(f"Hessian shape: {H.shape}")
print(f"Jacobian shape: {J.shape}")


