import numpy as np
import torch

from torch import linalg as LA
from scipy import optimize


class SCQP_Optimizer():
    '''
    Spherically constrained quadratic programming solver.
    '''
    def __init__(self):
        self.b = None
        self.q = None
        self.c = None
        self.s = None
        self.indices = None


    def solve_diag(self, q, b):
        '''
        Minimizes problem
            f(x) = 1/2 * x.T @ diag(q) @ x + b.T @ x
            s.t. ||x|| = 1

        Parameters
        ----------
        b : torch.tensor
            1D-vector, non-zero
        q : torch.tensor
            1D-vector, 0 < q[i] <= q[i + 1] for all i.

        Output
        ----------
        x : torch.tensor
            optimal value
        '''
        s, c = 1 + (q - q[0]) / torch.norm(b), b / torch.norm(b)
        s_, counts = np.unique(s.round(decimals=6).cpu(), return_counts=True)
        inds = np.concatenate(([0], np.cumsum(counts)))
        c_grid = [c[inds[j]:inds[j+1]] for j in range(s_.shape[0])]
        c_ = torch.tensor([torch.norm(segment) for segment in c_grid])
        z = self.solve_diag_unique(s_, c_)
        res =  torch.concat(
            [c_grid[j] * z[j] / c_[j] for j in range(s_.shape[0])]
        )
        return res


    def solve_diag_unique(self, q, b):
        self.q = q
        self.b = b
        self.indices = np.concatenate(([0], b[1:].nonzero().T[0] + 1))
        s_, c_ = q[self.indices], b[self.indices]
        if b[0] == 0:
            pos = c_[1:] / (1 - s_[1:])
            d = torch.inner(pos, pos)
            if d < 1 and s_[0] < s_[1]:
                return self.solve_particular_case(pos, d)
            else:
                x = self.solve_diag_nnz(s_[1:] + (1 - s_[1]), c_[1:])
                self.complete_solution(torch.concat([0],x))
        x = self.solve_diag_nnz(s_, c_)
        return self.complete_solution(x)


    def solve_diag_nnz(self, s, c):
        self.s = s
        self.c = c
        lam = self.find_lam()
        return c / (lam - s)

    def find_lam(self):
        sol = optimize.root_scalar(
            lambda x : self.lam_oracle(x),
            bracket=self.find_bounds(),
        )
        assert(sol.converged)
        return sol.root

    def lam_oracle(self, lam):
        return 1 - (torch.norm(self.c / (lam - self.s)) ** 2)

    def find_bounds(self):
        t_1 = self.root(self.c[0].item(), self.s[1] - self.s[0])
        t_2 = self.root(self.c[0].item(), self.s[-1] - self.s[0])
        return [1 - t_1, 1 - t_2]

    def root(self, c, d):
        polynomial = self.coefs(c, d)
        roots = np.roots(polynomial)
        condition = (roots >= np.abs(c)) & (roots < 1) & (abs(roots.imag)<1e-6)
        return roots[condition][0].real

    def coefs(self, c, d):
        return [
            1,
            2*d,
            d**2 - 1,
            -2 * c**2 * d,
            -c**2 * d**2,
        ]


    def solve_particular_case(self, pos, d):
        x = torch.zeros(pos.shape)
        x[0] = torch.sqrt(1 - d)
        x[1:] = pos
        return self.complete_solution(x, check_sign=True)


    def complete_solution(self, x, check_sign=False):
        new_x = torch.zeros(self.b.shape)
        new_x[self.indices] = x
        if check_sign:
            alt_x = new_x
            alt_x[0] = -alt_x[0]
            if self.func(new_x) > self.func(alt_x):
                return alt_x
        return new_x


    def func(self, x):
        return torch.inner(x, self.q * x) / 2 + torch.inner(self.b, x)



class Linreg_Optimizer():
    '''
    Linear regression with bound constraint solver.
    '''

    def solve(self, A, y, delta, repeats=1):
        '''
        Minimizes
            f(x) = ||x||^2
            s.t. ||Ax - y|| <= delta

        Parameters
        ----------
        A : torch.tensor
            Regressor matrix. N_1 >= N_2.
        y : torch.tensor
            Vector of dependent variables
        delta: float
            Bound on the regression error. Must be nonnegative.

        Output
        ----------
        x :  torch.tensor
            optimal value
        '''
        u, s, vh = LA.svd(A, full_matrices=False)
        y_ = torch.concat([u.T @ y[u.shape[0] * j:u.shape[0] * (j+1)] for j in range(repeats)])
        comp = y - torch.concat([u @ y_[u.shape[1] * j:u.shape[1] * (j+1)] for j in range(repeats)])
        assert(delta < torch.norm(y))
        assert(delta >= torch.norm(comp))
        delta_  = torch.sqrt(delta**2 - torch.inner(comp, comp))
        s =  torch.concat([s for _ in range(repeats)])
        q = delta_ / s**2
        b = -y_ / s**2
        z = SCQP_Optimizer().solve_diag(q, b)
        res = (y_ - delta_ * z) / s
        return torch.concat([vh.T @ res[vh.shape[0] * j:vh.shape[0] * (j+1)] for j in range(repeats)])


    def solve_matrix(self, A, B, delta):
        '''
        Minimizes
            f(X) = ||X||_F
            s.t. ||X @ A - B|| <= delta

        Parameters
        ----------
        A, B : torch.tensor
            2D-arrays.
        delta: float
            Bound on the regression error. Must be nonnegative.

        Output
        ----------
        X :  torch.tensor
            optimal value
        '''
        output_shape = (B.shape[0], A.shape[0])
        x_vec = self.solve(A.T, B.flatten(), delta, repeats=B.shape[0])
        return x_vec.reshape(output_shape)