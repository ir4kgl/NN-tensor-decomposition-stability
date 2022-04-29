import torch
from torch import trace

class SS_Regularizer_2D_Base():
    '''
    Base class for implementation of sensitivity regularizer for model
    with convolutional layer represented as instance of Tens_Conv_2D_Base subclass.

    Parameters
    ----------
        layer : string
            model's layer with required block

        index : int
            index of a block with required convolutional layer

        conv_layer : string
            name of required convolutional layer
    '''
    def __init__(self, layer, index, conv_layer, regcoef=1.):
        self.layer = layer
        self.index = index
        self.conv_layer = conv_layer
        self.regcoef = regcoef

    def get_layer(self, model):
        return getattr(getattr(model, self.layer)[self.index], self.conv_layer)

    def calc_penalty(self, model):
        '''
        Returns penalty based on sensitivity function.
        '''
        return self.calc_sensitivity(model).item() * self.regcoef

    def calc_sensitivity(self, model):
        layer = self.get_layer(model)
        n1, n2, n3 = layer.size()
        t1, t2, t3 = self.calc_terms(*layer.get_factors())
        return t1 * n1 + t2 * n2 + t3 * n3

    def calc_terms(self, A, B, C):
        '''
        Calculates auxiliary terms for sensitivity function.

        Parameters
        ----------
        A, B, C : torch.tensor
            factors of tensor decomposition.
        '''
        raise NotImplementedError


class CPD_Sensitivity_Regularizer_2D(SS_Regularizer_2D_Base):
    def calc_terms(self, A, B, C):
        At_A = A.T @ A
        Bt_B = B.T @ B
        Ct_C = C.T @ C
        t1 = trace(Bt_B * Ct_C)
        t2 = trace(At_A * Ct_C)
        t3 = trace(At_A * Bt_B)
        return t1, t2, t3


class TKD_Sensitivity_Regularizer_2D(SS_Regularizer_2D_Base):
    def calc_terms(self, A, B, C):
        t1 = torch.einsum('abc,abd,ec,ed', B, B, C, C)
        t2 = torch.norm(A) * torch.norm(C)
        t3 = torch.einsum('abc,adc,eb,ed', B, B, A, A)
        return t1, t2, t3


class TC_Sensitivity_Regularizer_2D(SS_Regularizer_2D_Base):
    def calc_terms(self, A, B, C):
        t1 = torch.einsum('abc,abd,efc,efd', B, B, C, C)
        t2 = torch.einsum('abc,abd,efc,efd', A, A, C, C)
        t3 = torch.einsum('abc,abd,ecf,edf', A, A, B, B)
        return t1, t2, t3
