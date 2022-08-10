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

    def factorize(K, rank, correct=False, correction_args=None):
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
        correction_args : dict
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
    def factorize(self, K, rank, correct=False, correction_args=None):
        self.K = K.reshape((K.shape[0],K.shape[1],-1)).permute((0,2,1))
        self.K = self.K.detach().numpy()
        self.factors = decomposition.parafac(self.K, rank).factors
        if correct:
            self.correct(**correction_args)
        return self.factors

    def correct(self, delta, n_iters):
        A, B, C = self.factors
        for _ in range(n_iters):
            A = self.optimization_step(self.K, A, B, C, delta)
            B = self.optimization_step(self.K.transpose((1,0,2)), B, A, C, delta)
            C = self.optimization_step(self.K.transpose((2,0,1)), C, A, B, delta)
        self.factors =  torch.tensor(A), torch.tensor(B), torch.tensor(C)

    def optimization_step(self, K, A, B, C, delta):
        scaler = self.find_scaler(B, C)
        regressor = LA.khatri_rao(B, C).T / scaler[:, None]
        target = K.reshape((K.shape[0],-1), order='C')
        A_raw = Linreg_Optimizer().solve_matrix(regressor, target, delta)
        return A_raw * scaler

    def find_scaler(self, B, C):
        return np.sqrt(
            LA.norm(B, axis=0)**2 + LA.norm(C, axis=0)**2
        )
