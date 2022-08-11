import numpy as np
from optimization import Linreg_Optimizer
from scipy import linalg as LA

from tensorly import decomposition


class Tens_Factorizer():
    '''
    Base class for tensor factorization.
    '''
    def __init__(self):
        self.K = None
        self.factors = None
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
    def factorize(self, K, rank, correct=False, args=None):
        self.K = K
        self.factors = decomposition.parafac(K, rank).factors
        if correct:
            self.delta = LA.norm(self.K - self.contract())
            self.correct(**args)
        return self.factors

    def correct(self, n_iters):
        for i in range(n_iters):
            for j in range(3):
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
        self.factors[0] = A_raw * scaler

    def find_scaler(self, B, C):
        return np.sqrt(
            LA.norm(B, axis=0)**2 + LA.norm(C, axis=0)**2
        )
