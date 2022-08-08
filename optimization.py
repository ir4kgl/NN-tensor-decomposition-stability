import numpy as np

from scipy import linalg as LA
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
        b : np.array
            1D-vector, non-zero
        q : np.array
            1D-vector, 0 < q[i] <= q[i + 1] for all i.

        Output
        ----------
        x :  np.array
            optimal value
        '''
        s, c = 1 + (q - q[0]) / LA.norm(b), b / LA.norm(b)
        s_, counts = np.unique(s.round(decimals=5), return_counts=True)
        inds = np.concatenate(([0], np.cumsum(counts)))
        c_grid = [c[inds[j]:inds[j+1]] for j in range(s_.shape[0])]
        c_ = np.array([LA.norm(segment) for segment in c_grid])
        z = self.solve_diag_unique(s_, c_)
        res =  np.concatenate(
            [c_grid[j] * z[j] / c_[j] for j in range(s_.shape[0])]
        )
        return res


    def solve_diag_unique(self, q, b):
        self.q = q
        self.b = b
        self.indices = np.concatenate(([0], b[1:].nonzero()[0] + 1))
        s_, c_ = q[self.indices], b[self.indices]
        if b[0] == 0:
            pos = c_[1:] / (1 - s_[1:])
            d = np.inner(pos, pos)
            if d < 1 and s_[0] < s_[1]:
                return self.solve_particular_case(pos, d)
            else:
                x = self.solve_diag_nnz(s_[1:] + (1 - s_[1]), c_[1:])
                self.complete_solution(np.concatenate([0],x))
        x = self.solve_diag_nnz(s_, c_)
        return self.complete_solution(x)


    def solve_diag_nnz(self, s, c):
        self.s = s
        self.c = c
        lam = self.find_lam()
        return c / (lam - s)

    def find_lam(self):
        sol = optimize.minimize_scalar(
            lambda x : self.lam_oracle(x),
            bounds=self.find_bounds(),
            method='bounded',
        )
        assert(sol.success)
        return sol.x

    def lam_oracle(self, x):
        signed_res = 1 - LA.norm(self.c / (x - self.s))
        return signed_res ** 2

    def find_bounds(self):
        t_1 = self.root(self.c[0], self.s[1] - self.s[0])
        t_2 = self.root(self.c[0], self.s[-1] - self.s[0])
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
        x = np.zeros(pos.shape)
        x[0] = np.sqrt(1 - d)
        x[1:] = pos
        return self.complete_solution(x, check_sign=True)


    def complete_solution(self, x, check_sign=False):
        new_x = np.zeros(self.b.shape)
        new_x[self.indices] = x
        if check_sign:
            alt_x = new_x
            alt_x[0] = -alt_x[0]
            if self.func(new_x) > self.func(alt_x):
                return alt_x
        return new_x


    def func(self, x):
        return np.inner(x, self.q * x) / 2 + np.inner(self.b, x)



class Linreg_Optimizer():
    '''
    Linear regression with bound constraint solver.
    '''

    def solve(self, A, y, delta):
        '''
        Minimizes
            f(x) = ||x||^2
            s.t. ||Ax - y|| <= delta

        Parameters
        ----------
        A : np.array
            Regressor matrix. N_1 >= N_2.
        y : np.array
            Vector of dependent variables
        delta: float
            Bound on the regression error. Must be nonnegative.

        Output
        ----------
        x :  np.array
            optimal value
        '''
        u, s, vh = LA.svd(A, full_matrices=False)
        y_ = u.T @ y
        comp = y - u @ y_
        delta_  = np.sqrt(delta**2 - np.inner(comp, comp))
        q = delta_ / s**2
        b = -y_ / s**2
        z = SCQP_Optimizer().solve_diag(q, b)
        return vh.T @ ( (y_ - delta_ * z) / s)


    def solve_matrix(self, A, B, delta):
        '''
        Minimizes
            f(X) = ||X||_F
            s.t. ||X @ A - B|| <= delta

        Parameters
        ----------
        A, B : np.array
            2D-arrays.
        delta: float
            Bound on the regression error. Must be nonnegative.

        Output
        ----------
        X :  np.array
            optimal value
        '''
        output_shape = (B.shape[0], A.shape[0])
        B_vec = B.flatten(order='C')
        A_new = LA.kron(A, np.eye(B.shape[0])).T
        x_vec = self.solve(A_new, B_vec, delta)
        return x_vec.reshape(output_shape, order='C')
