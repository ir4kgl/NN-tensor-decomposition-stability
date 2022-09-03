from optimization import Linreg_Optimizer

import torch
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
    def __init__(self, max_iters=20000):
        super().__init__()
        self.n_factors = 3
        self.max_iters = max_iters
        self.prev = None

    def factorize(self, K, rank, correct=False):
        self.K = K
        self.factors = parafac(K, rank, init='random', random_state=1).factors
        if correct:
            self.delta = torch.norm(self.K - self.contract())
            self.correct()
        return self.factors

    def correct(self):
        init = self.ss()
        for i in range(self.max_iters):
            for j in range(self.n_factors):
                self.optimization_step()
                self.cyclic_shift()
            cur_ss = self.ss()
            if self.prev is not None:
                if (self.prev - cur_ss) / init < 1e-5:
                    return
            self.prev = cur_ss

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
        regressor = (self.khatri_rao(B, C) / scaler).T
        target = self.K.reshape((self.K.shape[0], -1))
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = A_raw / scaler
        del A_raw
        torch.cuda.empty_cache()

    def find_scaler(self, B, C):
        n1, n2, n3 = self.K.shape
        return torch.sqrt(
            n3 * torch.norm(B, dim=0)**2 + n2 * torch.norm(C, dim=0)**2
        )

    def khatri_rao(self, A, B):
        assert(A.shape[1] == B.shape[1])
        out_shape = (A.shape[0] * B.shape[0], A.shape[1])
        res = torch.zeros(out_shape).cuda()
        for j in range(out_shape[1]):
            cur_col = torch.einsum('i,j->ij', A[:,j], B[:,j]).ravel()
            res[:, j] = cur_col
        return res

    def ss(self):
        A, B, C = self.factors
        n1, n2, n3 = self.K.shape
        A_norm = torch.norm(A, dim=0)**2
        B_norm = torch.norm(B, dim=0)**2
        C_norm = torch.norm(C, dim=0)**2
        t1 = torch.inner(B_norm, C_norm)
        t2 = torch.inner(A_norm, C_norm)
        t3 = torch.inner(A_norm, B_norm)
        return t1 * n1 + t2 * n2 + t3 * n3


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


class TC_Factorizer(Tens_Factorizer):
    def __init__(self, maxiters=10000):
        super().__init__()
        self.rank = None
        self.factors = None
        self.n_factors = 3
        self.old = None
        self.maxiters = maxiters
        self.prev = None

    def factorize(self, K, rank, n_iters_als, correct=False):
        self.K = K
        self.rank = rank
        self.als_decomposition(K, rank, n_iters_als)
        if correct:
            self.delta = torch.norm(self.K - self.contract())
            wandb.log({"error before correction": torch.norm(self.K - self.contract())})
            self.correct()
            wandb.log({"error after correction": torch.norm(self.K - self.contract())})
        return self.factors

    def contract(self):
        A, B, C = self.factors
        return torch.einsum('kai,ibj,jck->abc', A, B, C)

    def als_decomposition(self, K, rank, n_iters):
        n0, n1, n2 = self.K.size()
        r0, r1, r2 = self.rank
        gen = torch.Generator()
        gen.manual_seed(2147483647)
        self.factors = [
            torch.empty((r0,n0,r1)).normal_(generator=gen).cuda(),
            torch.empty((r1,n1,r2)).normal_(generator=gen).cuda(),
            torch.empty((r2,n2,r0)).normal_(generator=gen).cuda(),
        ]
        for i in range(n_iters):
            for j in range(self.n_factors):
                self.als_step(j)
                self.cyclic_shift()
            diff = torch.norm(self.K -self.contract()).item()
            if self.old is not None:
                if (self.old - diff) / self.old < 1e-5:
                    return self.factors
            self.old = diff


    def correct(self):
        init = self.ss()
        wandb.log({"sentivity": init})
        for i in range(self.maxiters):
            for j in range(self.n_factors):
                self.optimization_step()
                self.cyclic_shift()
            cur_ss = self.ss()
            wandb.log({"sentivity": cur_ss})
            if self.prev is not None:
                if (self.prev - cur_ss) / self.prev < 1e-5:
                    return
            self.prev = cur_ss

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

    def als_step(self, j):
        old = self.factors[0].clone()
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        regressor = torch.einsum('ibj,jck->bcik', B, C).reshape((-1, r0 * r1))
        target = self.K.reshape((self.K.shape[0], -1)).permute((1,0))
        self.factors[0] = LA.lstsq(regressor, target,)[0]
        self.factors[0] = self.factors[0].reshape((r1, r0, -1))
        self.factors[0] = self.factors[0].permute((1,2,0))
        del regressor
        del target
        torch.cuda.empty_cache()

    def optimization_step(self):
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        L = self.find_L()
        regressor = torch.einsum('ibj,jck->ikbc', B, C).reshape((r0 * r1, -1))
        regressor = LA.solve_triangular(L, regressor, upper=False)
        target = self.K.reshape((self.K.shape[0], -1))
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, self.delta)
        self.factors[0] = LA.solve_triangular(L.T, A_raw.T, upper=True)
        self.factors[0] = self.factors[0].reshape((r1, r0, -1))
        self.factors[0] = self.factors[0].permute((1,2,0))
        del A_raw
        del regressor
        del target
        del L
        torch.cuda.empty_cache()

    def find_L(self, core=False):
        A, B, C = self.factors
        r0, r1, r2 = self.rank
        B = B.reshape((r1, -1))
        C = C.permute((2,1,0)).reshape((r0, -1))
        W = torch.kron(B @ B.T, torch.eye(r0).cuda()) + torch.kron(torch.eye(r1).cuda(), C @ C.T)
        return LA.cholesky(W)

    def ss(self):
        A, B, C = self.factors
        n1, n2, n3 = self.K.shape
        t1 = torch.einsum('abc,abd,cfe,dfe', B, B, C, C)
        t2 = torch.einsum('abc,abd,cfe,dfe', C, C, A, A)
        t3 = torch.einsum('abc,abd,cfe,dfe', A, A, B, B)
        return t1 * n1 + t2 * n2 + t3 * n3