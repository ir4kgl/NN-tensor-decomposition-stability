import numpy as np
from optimization import Linreg_Optimizer
from scipy import linalg

from tensorly import decomposition


def find_scaler(B, C):
    return np.sqrt(
        linalg.norm(B, axis=0)**2 + linalg.norm(C, axis=0)**2
    )

def optimization_step(K, A, B, C, delta):
    scaler = find_scaler(B, C)
    regressor = linalg.khatri_rao(B, C).T / scaler[:, None]
    target = K.reshape((K.shape[0],-1), order='C')
    A_raw = Linreg_Optimizer().solve_matrix(regressor, target, delta)
    return A_raw * scaler

def correct_CPD(K, factors, delta=10, n_iters=5):
    '''
    Minimizes sensitivity of the CP-decomposition
    presersing approximation error delta.

    Parameters
    ----------
    K: np.array
        Given 3D-tensor
    factors: tuple of np.array
        Initial decomposition
    delta: float
        Approximation error

    Output
    ----------
    new_factors: tuple of np.array
        corrected decomposition
    '''
    A, B, C = factors
    for _ in range(n_iters):
        A = optimization_step(K, A, B, C, delta)
        B = optimization_step(K.transpose((1,0,2)), B, A, C, delta)
        C = optimization_step(K.transpose((2,0,1)), C, A, B, delta)
    return torch.tensor(A), torch.tensor(B), torch.tensor(C)


def factorize_CPD(K, rank, correct=False, correction_args=None):
    target = K.reshape((K.shape[0],K.shape[1],-1)).permute((0,2,1))
    target = target.detach().numpy()
    cp = decomposition.parafac(target, rank)
    factors = cp.factors
    if correct:
        factors = correct_CPD(target, factors, **correction_args)
    return factors
