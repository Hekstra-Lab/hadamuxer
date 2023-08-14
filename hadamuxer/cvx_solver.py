from hadamard import make_S_mat
import cvxpy as cvx
from pylab import *
from IPython import embed

n = 11

x_true = np.random.random((n, 1))
S = make_S_mat(n)
Sinv = np.linalg.inv(S)
k = np.random.poisson(S @ x_true).astype('float')


rate = cvx.Variable(x_true.shape)

ll = cvx.multiply(k, cvx.log(rate)) - rate
likelihood = cvx.sum(ll)


cons = [
    Sinv@rate >= 0.,
]

p = cvx.Problem(
    cvx.Maximize(likelihood),
    cons,
)



embed(colors='linux')
