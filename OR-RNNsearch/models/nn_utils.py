import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

epsilon = 1e-20

class MaskSoftmax(nn.Module):

    def __init__(self):

        super(MaskSoftmax, self).__init__()

    def forward(self, x, mask=None, dim=-1):

        # input torch tensor or variable, take max for numerical stability
        x_max = tc.max(x, dim=dim, keepdim=True)[0]
        x_minus = x - x_max
        x_exp = tc.exp(x_minus)
        if mask is not None: x_exp = x_exp * mask
        x = x_exp / ( tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon )

        return x

class MyLogSoftmax(nn.Module):

    def __init__(self, self_norm_alpha=None):

        super(MyLogSoftmax, self).__init__()
        self.sna = self_norm_alpha

    def forward(self, x, dim=-1):

        # input torch tensor
        x_max = tc.max(x, dim=dim, keepdim=True)[0]  # take max for numerical stability
        x_exp = tc.exp( x - x_max )
        x_exp_sum = tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon
        log_norm = tc.log( x_exp_sum ) + x_max
        x = x - log_norm    # get log softmax
        prob = x_exp / x_exp_sum

        # Sum_( log(P(xi)) - alpha * square( log(Z(xi)) ) )
        if self.sna is not None: x = x - self.sna * tc.pow(log_norm, 2)

        return log_norm, prob, x

'''Layer normalize the tensor x, averaging over the last dimension.'''
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(tc.ones(features))
        self.b_2 = nn.Parameter(tc.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    '''
        A two-layer Feed-Forward Network
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer of the FNN.
            droput(float): dropout probability(0-1.0).
    '''
    def __init__(self, input_size=512, filter_size=2048, output_size=512, dropout_prob=0.1):

        super(PositionwiseFeedForward, self).__init__()
        self.filter_transform = Linear(input_size, filter_size, bias=True)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.output_transform = Linear(filter_size, output_size, bias=True)

    def forward(self, x):

        # (batch_size, input_len, model_dim) -> (batch_size, input_len, model_dim)
        x = self.filter_transform(x)
        x = self.relu(x)

        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.output_transform(x)

        return x

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias is True:
        nn.init.constant_(m.bias, 0.)
    return m
