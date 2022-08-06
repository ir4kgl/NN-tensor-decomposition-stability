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

def correct_CPD(K, factors, delta, n_iters=5):
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
        C = optimization_step(K.transpose((2,0,1)), C, A, B, delta)
        B = optimization_step(K.transpose((1,0,2)), B, A, C, delta)
    return A, B, C


def factorize_CPD(K, rank, correct=False, correction_args=None):
    cp = decomposition.parafac(K, rank)
    factors = cp.factors
    if correct:
        factors = correct_CPD(K, factors, **correction_args)
    return factors
