import numpy as np
from optimization import Linreg_Optimizer
from scipy import linalg as LA

from tensorly.decomposition import parafac, partial_tucker


class Tens_Factorizer():
    '''
    Base class for tensor factorization.
    '''
    def __init__(self):
        self.K = None
        self.delta = None

    def factorize(K, rank, correct=False, args=None):
        '''
        Finds tensor decomposition of given tensor.

        Parameters
        ----------
        K : torch.tensor
            given tensor
        rank : int or tuple of ints
            rank of tensor decomposition
        correct : bool
            whether to use the correction method (minimize sensitivity)
        args : dict
            correction algorithm arguments. None if correct=False.

        Output
        ----------
        factors : tuple of torch.tensor
            factors of the tensor decompotion
        '''
        raise NotImplementedError

    def correct(delta, n_iters):
        '''
        Minimizes sensitivity of the decomposition
        presersing approximation error delta.

        Parameters
        ----------
        K: np.array
            given tensor
        factors: tuple of np.array
            initial decomposition
        delta: float
            approximation error

        Output
        ----------
        new_factors: tuple of np.array
            corrected decomposition
        '''
        raise NotImplementedError


class CPD_Factorizer(Tens_Factorizer):
    def __init__(self):
        super().__init__()
        self.factors = None
        self.n_factors = 3

    def factorize(self, K, rank, correct=False, args=None):
        self.K = K
        self.factors = parafac(K, rank).factors
        if correct:
            self.delta = LA.norm(self.K - self.contract())
            self.correct(**args)
        return self.factors

    def correct(self, n_iters):
        for i in range(n_iters):
            for j in range(self.n_factors):
                self.optimization_step()
                self.cyclic_shift()

    def cyclic_shift(self):
        self.K = self.K.transpose((2,0,1))
        self.factors = [
            self.factors[2],
            self.factors[0],
            self.factors[1]
        ]

    def contract(self):
        A, B, C = self.factors
        return np.einsum('ia,ja,ka->ijk', A, B, C)

    def optimization_step(self):
        A, B, C = self.factors
        scaler = self.find_scaler(B, C)
        regressor = LA.khatri_rao(B, C).T / scaler[:, None]
        target = self.K.reshape((self.K.shape[0], -1), order='C')
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = A_raw / scaler

    def find_scaler(self, B, C):
        return np.sqrt(
            LA.norm(B, axis=0)**2 + LA.norm(C, axis=0)**2
        )


class TKD_Factorizer(Tens_Factorizer):
    def __init__(self):
        super().__init__()
        self.core = None
        self.rank = None
        self.factors = None
        self.n_factors = 2

    def factorize(self, K, rank, correct=False, args=None):
        self.K = K
        self.rank = rank
        self.core, self.factors = partial_tucker(self.K, modes=(0,2), rank)
        if correct:
            self.delta = LA.norm(self.K - self.contract())
            self.correct(**args)
        return self.core, self.factors

    def correct(self, n_iters):
        for i in range(n_iters):
            for j in range(self.n_factors):
                self.optimization_step_factors()
                self.factors_shift()
            self.optimization_step_core()

    def optimization_step_core(self):
        L = self.find_L(core=True)
        regressor = LA.kron(self.factors[0], self.factors[1]).T
        regressor = LA.solve_triangular(L, regressor, lower=True)
        target = self.K.transpose(1,0,2).reshape((self.K.shape[1], -1), order='C')
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.core = LA.solve_triangular(L.T, A_raw.T)
        self.core = self.core.reshape(self.rank[0], self.rank[1], -1, order='C')
        self.core = self.core.transpose((0,2,1))

    def optimization_step_factors(self):
        B, C = self.factors
        L = self.find_L()
        regressor = np.einsum('ijc,kc->ijk', self.core, C).reshape((self.rank[0], -1), order='C')
        regressor = LA.solve_triangular(L, regressor, lower=True)
        target = self.K.reshape((self.K.shape[0], -1), order='C')
        B_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = LA.solve_triangular(L.T, B_raw.T).T

    def factors_shift(self):
        self.K = self.K.transpose((2,1,0))
        self.core = self.core.transpose((2,1,0))
        self.factors = [
            self.factors[1],
            self.factors[0]
        ]
        self.rank = [
            self.rank[1],
            self.rank[0]
        ]

    def contract(self):
        A = self.core
        B, C = self.factors
        return np.einsum('bjc,ib,kc->ijk', A, B, C)

    def find_L(self, core=False):
        B, C = self.factors
        r0, r1 = self.rank
        W = None
        if core:
            W = LA.kron(B.T @ B, np.eye(r1)) + LA.kron(np.eye(r0), C.T @ C)
        else:
            A = self.core.reshape((self.rank[0], -1))
            W = A @ A.T + np.eye(self.rank[0])
        return LA.cholesky(W, lower=True)


class TC_factorizer(Tens_Factorizer):
    def __init__(self):
        super().__init__()
        self.rank = None
        self.factors = None
        self.n_factors = 3

    def factorize(self, K, rank, correct=False, args=None):
        self.K = K
        self.rank = rank
        self.factors = tensor_ring(K, rank).factors
        if correct:
            self.delta = LA.norm(self.K - self.contract())
            self.correct(**args)
        return self.factors

    def contract(self):
        A, B, C = self.factors
        return np.einsum('kai,ibj,jck->abc', A, B, C)

    def correct(self, n_iters):
        for i in range(n_iters):
            for j in range(self.n_factors):
                self.optimization_step()
                self.cyclic_shift()

    def cyclic_shift(self):
        self.K = self.K.transpose((2,0,1))
        self.factors = [
            self.factors[2],
            self.factors[0],
            self.factors[1]
        ]
        self.rank = [
            self.rank[2],
            self.rank[0],
            self.rank[1],
        ]

    def optimization_step(self):
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        L = self.find_L()
        regressor = np.einsum('ibj,jck->ikbc', B, C).reshape((r0 * r2, -1), order='C')
        regressor = LA.solve_triangular(L, regressor, lower=True)
        target = self.K.reshape((self.K.shape[0], -1), order='C')
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = LA.solve_triangular(L.T, A_raw.T)
        self.factors[0] = self.factors[0].reshape((r0, r2, -1), order='C')
        self.factors[0] = self.factors[0].transpose((1,2,0))

    def find_L(self, core=False):
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        B = B.reshape((r0, -1), order='C')
        C = C.transpose(2,1,0).reshape((r2, -1), order='C')
        W = LA.kron(B @ B.T, np.eye(r2)) + LA.kron(np.eye(r0), C @ C.T)
        return LA.cholesky(W, lower=True)
