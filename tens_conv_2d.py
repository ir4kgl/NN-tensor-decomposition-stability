import torch
from torch import nn
from functools import reduce

def replace_conv_layer_2D(model, layer, tn, tn_args, device):
    '''
    Replace given convolutional layer with tensor decomposition.

    Parameters
    ----------
    model :
        model to compress

    layer : string
        model's convolutional layer to compress

    tn : Tens_Conv_2D_Base subclass
        class of tensor factorized convolutional layer

    tn_args : dict
        tn's constructor arguments

    device : torch.device
    '''
    submodule_names = layer.split(sep='.')
    last_block = reduce(getattr, submodule_names[:-1], model)
    layer_name = submodule_names[-1]
    new_value = tn(getattr(last_block, layer_name), **tn_args).to(device)
    setattr(last_block, layer_name, new_value)


class Tens_Conv_2D_Base(nn.Module):
    '''
    Base class for 2D convolutional kernel represented as tensor decomposition.

    Parameters
    ----------
        orig_layer :
            original convolutional layer to replace with a tensor decomposition.

        rank : int or tuple of ints
            rank of tensor decomposition.

    '''
    def __init__(self, orig_layer, rank):
        super().__init__()
        self.rank = rank
        self.in_channels = orig_layer.in_channels
        self.out_channels = orig_layer.out_channels
        self.kernel_size = orig_layer.kernel_size
        self.padding = orig_layer.padding
        self.stride = orig_layer.stride
        self.bias = orig_layer.bias

    def size(self):
        '''
        Returns shape of the represented kernel.
        '''
        return (
            self.in_channels,
            self.kernel_size[0] * self.kernel_size[1],
            self.out_channels,
        )

    def forward(self, x):
        raise NotImplementedError

    def calc_penalty(self):
        '''
        Calculates sensitivity of the factorized layer.
        '''
        n1, n2, n3 = self.size()
        t1, t2, t3 = self.calc_terms(*self.get_factors())
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

    def get_factors(self):
        '''
        Returns factors of tensor decomposition.
        '''
        raise NotImplementedError

    def set_factors(self, factors):
        raise NotImplementedError


class CPD_Conv_2D(Tens_Conv_2D_Base):
    def __init__(self, orig_layer, rank, factors=None):
        super().__init__(orig_layer, rank)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.rank,
                kernel_size=(1, 1),
                bias=False),

            nn.Conv2d(
                in_channels=self.rank,
                out_channels=self.rank,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.rank,
                bias=False),

            nn.Conv2d(
                in_channels=self.rank,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                bias=self.bias),
        )

        self.shapes = (
            self.layers[0].weight.shape,
            self.layers[1].weight.shape,
            self.layers[2].weight.shape
        )

        if factors is not None:
            self.set_factors(factors)


    def forward(self, x):
        return self.layers(x)

    def get_factors(self):
        A = self.layers[0].weight
        B = self.layers[1].weight
        C = self.layers[2].weight
        return (
            A.reshape((self.rank, -1)).permute((1, 0)),
            B.reshape((self.rank, -1)).permute((1, 0)),
            C.reshape((-1, self.rank)),
        )

    def calc_terms(self, A, B, C):
        A_norm = torch.norm(A, dim=0)**2
        B_norm = torch.norm(B, dim=0)**2
        C_norm = torch.norm(C, dim=0)**2
        t1 = torch.inner(B_norm, C_norm)
        t2 = torch.inner(A_norm, C_norm)
        t3 = torch.inner(A_norm, B_norm)
        return t1, t2, t3

    def set_factors(self, factors):
        A, B, C = factors
        A = A.permute((1, 0)).reshape(self.shapes[0])
        B = B.permute((1, 0)).reshape(self.shapes[1])
        C = C.reshape(self.shapes[2])

        with torch.no_grad():
            self.layers[0].weight = nn.Parameter(A)
            self.layers[1].weight = nn.Parameter(B)
            self.layers[2].weight = nn.Parameter(C)


class TKD_Conv_2D(Tens_Conv_2D_Base):
    def __init__(self, orig_layer, rank):
        super().__init__(orig_layer, rank)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.rank[0],
                kernel_size=(1, 1),
                bias=False),

            nn.Conv2d(
                in_channels=self.rank[0],
                out_channels=self.rank[1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False),

            nn.Conv2d(
                in_channels=self.rank[1],
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                bias=self.bias),
        )
        self.shapes = (
            (self.rank[0], self.in_channels),
            (self.rank[1], self.rank[0], -1),
            (self.out_channels, self.rank[1]),
        )
        self.permutations = (
            (1, 0),
            (2, 1, 0),
            None,
        )

    def forward(self, x):
        return self.layers(x)

    def get_factors(self):
        return (
            self.layers[0].weight.reshape(self.shapes[0]).permute(self.permutations[0]),
            self.layers[1].weight.reshape(self.shapes[1]).permute(self.permutations[1]),
            self.layers[2].weight.reshape(self.shapes[2]),
        )

    def calc_terms(self, A, B, C):
        t1 = torch.einsum('abc,abd,ec,ed', B, B, C, C)
        t2 = (torch.norm(A) * torch.norm(C)) ** 2
        t3 = torch.einsum('abc,adc,eb,ed', B, B, A, A)
        return t1, t2, t3

    # def set_factors(self, factors):


class TC_Conv_2D(Tens_Conv_2D_Base):
    def __init__(self, orig_layer, rank):
        super().__init__(orig_layer, rank)
        self.in_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.rank[0] * self.rank[1],
            kernel_size=(1, 1),
            bias=False)
        self.mid_conv = nn.Conv3d(
            in_channels=self.rank[1],
            out_channels=self.rank[2],
            kernel_size=(1, *self.kernel_size),
            stride=(1, *self.stride),
            padding=(0, *self.padding),
            bias=False)
        self.out_conv = nn.Conv2d(
            in_channels=self.rank[2]*self.rank[0],
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            bias=self.bias)
        self.shapes = (
            (self.rank[0], self.rank[1], self.in_channels),
            (-1, self.rank[1], self.rank[2]),
            (self.out_channels, self.rank[0], self.rank[2]),
        )
        self.permutations = (
            (2, 0, 1),
            (2, 3, 4, 1, 0),
            None,
        )

    def forward(self, x):
        N, _, H, W = x.shape
        H_out, W_out = int(H / self.stride[0]), int(W / self.stride[1])
        y = self.in_conv(x)
        y = y.reshape(N, self.rank[0], self.rank[1], H, W)
        y = y.permute(0, 2, 1, 3, 4)
        y = self.mid_conv(y)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(N, self.rank[2] * self.rank[0], H_out, W_out)
        return self.out_conv(y)

    def get_factors(self):
        return (
            self.in_conv.weight.reshape(self.shapes[0]).permute(self.permutations[0]),
            self.mid_conv.weight.permute(self.permutations[1]).reshape(self.shapes[1]),
            self.out_conv.weight.reshape(self.shapes[2]),
        )

    def calc_terms(self, A, B, C):
        t1 = torch.einsum('abc,abd,efc,efd', B, B, C, C)
        t2 = torch.einsum('abc,abd,efc,efd', A, A, C, C)
        t3 = torch.einsum('abc,abd,ecf,edf', A, A, B, B)
        return t1, t2, t3

    # def set_factors(self, factors):
