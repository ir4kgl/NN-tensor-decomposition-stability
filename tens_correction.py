# from optimization import Linreg_Optimizer

from torch import linalg as LA

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from tensorly.tenalg import khatri_rao

tl.set_backend('pytorch')


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
        K: torch.tensor
            given tensor
        factors: tuple of torch.tensor
            initial decomposition
        delta: float
            approximation error

        Output
        ----------
        new_factors: tuple of torch.tensor
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
            self.delta = torch.norm(self.K - self.contract(), p=2)
            self.correct(**args)
        return self.factors

    def correct(self, n_iters):
        for i in range(n_iters):
            for j in range(self.n_factors):
                self.optimization_step()
                self.cyclic_shift()

    def cyclic_shift(self):
        self.K = self.K.permute((2,0,1))
        self.factors = [
            self.factors[2],
            self.factors[0],
            self.factors[1]
        ]

    def contract(self):
        A, B, C = self.factors
        return torch.einsum('ia,ja,ka->ijk', A, B, C)

    def optimization_step(self):
        A, B, C = self.factors
        scaler = self.find_scaler(B, C)
        regressor = self.khatri_rao(B, C).T / scaler[:, None]
        target = self.K.reshape((self.K.shape[0], -1))
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = A_raw / scaler

    def find_scaler(self, B, C):
        return torch.sqrt(
            torch.norm(B, dim=0, p=2)**2 + torch.norm(C, dim=0, p=2)**2
        )

    def khatri_rao(self, A, B):
        out_shape = (A.shape[0] * B.shape[0], A.shape[1])
        res = torch.zeros(out_shape).cuda()
        for j in range(out_shape[1]):
            cur_col = torch.einsum('i,j->ij', A[:,j], B[:,j]).ravel()
            res[:, j] = cur_col
        return res


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
        self.core, self.factors = partial_tucker(self.K, modes=(0,2), rank=rank)
        if correct:
            self.delta = torch.norm(self.K - self.contract())
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
        regressor = torch.kron(self.factors[0], self.factors[1]).T
        regressor = LA.solve_triangular(L, regressor, upper=False)
        target = self.K.permute((1,0,2)).reshape((self.K.shape[1], -1))
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.core = LA.solve_triangular(L.T, A_raw.T, upper=True)
        self.core = self.core.reshape(self.rank[0], self.rank[1], -1)
        self.core = self.core.permute((0,2,1))

    def optimization_step_factors(self):
        B, C = self.factors
        L = self.find_L()
        regressor = torch.einsum('ijc,kc->ijk', self.core, C).reshape((self.rank[0], -1))
        regressor = LA.solve_triangular(L, regressor, upper=False)
        target = self.K.reshape((self.K.shape[0], -1))
        B_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = LA.solve_triangular(L.T, B_raw.T, upper=True).T

    def factors_shift(self):
        self.K = self.K.permute((2,1,0))
        self.core = self.core.permute((2,1,0))
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
        return torch.einsum('bjc,ib,kc->ijk', A, B, C)

    def find_L(self, core=False):
        B, C = self.factors
        r0, r1 = self.rank
        W = None
        if core:
            W = torch.kron(B.T @ B, torch.eye(r1).cuda()) + torch.kron(torch.eye(r0).cuda(), C.T @ C)
        else:
            A = self.core.reshape((self.rank[0], -1))
            W = A @ A.T + torch.eye(self.rank[0]).cuda()
        return LA.cholesky(W)


class TC_factorizer(Tens_Factorizer):
    def __init__(self):
        super().__init__()
        self.rank = None
        self.factors = None
        self.n_factors = 3

    def factorize(self, K, rank, n_iters_als, correct=False, args=None):
        self.K = K
        self.rank = rank
        self.als_decomposition(K, rank, n_iters_als)
        if correct:
            self.delta = torch.norm(self.K - self.contract(), p=2)
            self.correct(**args)
        return self.factors

    def contract(self):
        A, B, C = self.factors
        return torch.einsum('kai,ibj,jck->abc', A, B, C)

    def als_decomposition(self, K, rank, n_iters):
        n0, n1, n2 = self.K.size()
        r0, r1, r2 = self.rank
        self.factors = [
            torch.empty((r2,n0,r0)).normal_().cuda(),
            torch.empty((r0,n1,r1)).normal_().cuda(),
            torch.empty((r1,n2,r2)).normal_().cuda(),
        ]
        for i in range(n_iters):
            for j in range(self.n_factors):
                self.als_step()

    def correct(self, n_iters):
        for i in range(n_iters):
            for j in range(self.n_factors):
                self.optimization_step()
                self.cyclic_shift()

    def cyclic_shift(self):
        self.K = self.K.permute((2,0,1))
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

    def als_step(self):
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        regressor = torch.einsum('ibj,jck->bcik', B, C).reshape((-1, r0 * r2))
        target = self.K.reshape((self.K.shape[0], -1)).permute((1,0))
        self.factors[0] = LA.lstsq(regressor, target)[0]
        self.factors[0] /= self.factors[0].max(0, keepdim=True)[0]
        self.factors[0] = self.factors[0].reshape((r0, r2, -1))
        self.factors[0] = self.factors[0].permute((1,2,0))

    def optimization_step(self):
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        L = self.find_L()
        regressor = torch.einsum('ibj,jck->ikbc', B, C).reshape((r0 * r2, -1))
        regressor = LA.solve_triangular(L, regressor, upper=False)
        target = self.K.reshape((self.K.shape[0], -1))
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = LA.solve_triangular(L.T, A_raw.T)
        self.factors[0] = self.factors[0].reshape((r0, r2, -1))
        self.factors[0] = self.factors[0].permute((1,2,0))

    def find_L(self, core=False):
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        B = B.reshape((r0, -1))
        C = C.permute((2,1,0)).reshape((r2, -1))
        W = torch.kron(B @ B.T, torch.eye(r2).cuda()) + torch.kron(torch.eye(r0).cuda(), C @ C.T)
        return LA.cholesky(W)
