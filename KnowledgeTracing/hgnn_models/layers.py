import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class HGNN_conv(nn.Module): # Inherited from module
    #   in_features: size of each input sample
    #   out_features: size of each output sample
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        # Convert a non trainable type Tensor to trainable type Parameter
        # Parameter definition
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        # Parameter initialization function
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # forward function
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

