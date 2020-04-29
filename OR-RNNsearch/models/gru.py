import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

# inheriting from nn.Module
class GRU(nn.Module):

    '''
    Gated Recurrent Unit network with initial state as parameter:

        z_t = sigmoid((x_t dot W_xz + b_xz) + (h_{t-1} dot U_hz + b_hz))
        r_t = sigmoid((x_t dot W_xr + b_xr) + (h_{t-1} dot U_hr + b_xr))

        => zr_t = sigmoid((x_t dot W_xzr + b_xzr) + (h_{t-1} dot U_hzr + b_hzr))
        slice ...

        h_above = tanh(x_t dot W_xh + b_xh + (h_{t-1} dot U_hh + b_hh) * r_t)

        #h_t = (1 - z_t) * h_above + z_t * h_{t-1}
        h_t = (1 - z_t) * h_{t-1} + z_t * h_above

    all parameters are initialized in [-0.01, 0.01]
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 enc_hid_size=None,
                 with_ln=False,
                 dropout_prob=0.1,
                 prefix='GRU', **kwargs):

        # input vector size and hidden vector size
        # calls the init function of nn.Module
        super(GRU, self).__init__()

        self.enc_hid_size = enc_hid_size
        self.hidden_size = hidden_size
        self.with_ln = with_ln
        self.prefix = prefix

        self.xh = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.xrz = nn.Linear(input_size, 2 * hidden_size)
        self.hrz = nn.Linear(hidden_size, 2 * hidden_size)

        if self.enc_hid_size is not None:
            self.crz = nn.Linear(enc_hid_size, 2 * hidden_size)
            self.ch = nn.Linear(enc_hid_size, hidden_size)

        #if self.with_ln is not True:

            #self.xz = nn.Linear(input_size, hidden_size)
            #self.hz = nn.Linear(hidden_size, hidden_size)
            #self.xr = nn.Linear(input_size, hidden_size)
            #self.hr = nn.Linear(hidden_size, hidden_size)

            #if self.enc_hid_size is not None:
                #self.cz = nn.Linear(2 * enc_hid_size, hidden_size)
                #self.cr = nn.Linear(2 * enc_hid_size, hidden_size)
                #self.ch = nn.Linear(2 * enc_hid_size, hidden_size)

        if self.with_ln is True:

            self.ln0 = nn.LayerNorm(2 * hidden_size)
            self.ln1 = nn.LayerNorm(2 * hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.ln3 = nn.LayerNorm(hidden_size)

            if self.enc_hid_size is not None:
                self.ln4 = nn.LayerNorm(2 * hidden_size)
                self.ln5 = nn.LayerNorm(hidden_size)

        if dropout_prob is not None and 0. < dropout_prob <= 1.0:
            self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob

    '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
    '''
    def forward(self, x_t, x_m, h_tm1, attend=None):

        x_rz_t, h_rz_tm1, x_h_t = self.xrz(x_t), self.hrz(h_tm1), self.xh(x_t)

        if self.with_ln is not True:

            if self.enc_hid_size is None:
                #r_t = tc.sigmoid(self.xr(x_t) + self.hr(h_tm1))
                #z_t = tc.sigmoid(self.xz(x_t) + self.hz(h_tm1))
                #h_t_above = tc.tanh(self.xh(x_t) + self.hh(r_t * h_tm1))
                rz_t = x_rz_t + h_rz_tm1
            else:
                #z_t = tc.sigmoid(self.xz(x_t) + self.hz(h_tm1) + self.cz(attend))
                #r_t = tc.sigmoid(self.xr(x_t) + self.hr(h_tm1) + self.cr(attend))
                #h_t_above = tc.tanh(self.xh(x_t) + self.hh(r_t * h_tm1) + self.ch(attend))
                a_rz_t, a_h_t = self.crz(attend), self.ch(attend)
                rz_t = x_rz_t + h_rz_tm1 + a_rz_t

        else:

            x_rz_t, h_rz_tm1, x_h_t = self.ln0(x_rz_t), self.ln1(h_rz_tm1), self.ln2(x_h_t)

            if self.enc_hid_size is None:
                rz_t = x_rz_t + h_rz_tm1
            else:
                a_rz_t, a_h_t = self.crz(attend), self.ch(attend)
                a_rz_t, a_h_t = self.ln4(a_rz_t), self.ln5(a_h_t)
                rz_t = x_rz_t + h_rz_tm1 + a_rz_t

        assert rz_t.dim() == 2
        rz_t = tc.sigmoid(rz_t)
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]

        h_h_tm1 = self.hh(r_t * h_tm1)
        if self.with_ln: h_h_tm1 = self.ln3(h_h_tm1)
        #h_h_tm1 = h_h_tm1 * r_t
        if self.enc_hid_size is None: h_h_tm1 = x_h_t + h_h_tm1
        else: h_h_tm1 = x_h_t + h_h_tm1 + a_h_t

        h_t_above = tc.tanh(h_h_tm1)

        if self.dropout_prob is not None and 0. < self.dropout_prob <= 1.0:
            h_t_above = self.dropout(h_t_above)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above

        if x_m is not None:
            #h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
            h_t = x_m[:, None] * h_t

        return h_t

'''
Linear Linear Transformation enhanced GRU with initial state as parameter:

    z_t = sigmoid( h_{t-1} dot U_hz + b_hz )
    r_t = sigmoid( h_{t-1} dot U_hr + b_xr )

    => zr_t = sigmoid( h_{t-1} dot U_hzr + b_hzr )
    slice ...

    l_t = sigmoid( ( x_t dot W_xl + b_xl ) + ( h_{t-1} dot U_hl + b_hl ) )

    h_above = tanh( ( x_t dot W_xh + b_xh ) + r_t * ( h_{t-1} dot U_hh + b_hh ) ) + l_t * H(x_t)

    h_t = (1 - z_t) * h_{t-1} + z_t * h_above

all parameters are initialized in [-0.01, 0.01]
'''
class TransiLNCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_prob=None, prefix='TransiLNCell', **kwargs):

        super(TransiLNCell, self).__init__()

        self.hidden_size = hidden_size
        self.prefix = prefix

        self.xrz = nn.Linear(input_size, 2 * hidden_size, bias=False)
        self.hrz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

        self.xl = nn.Linear(input_size, hidden_size, bias=False)
        self.hl = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wx = nn.Linear(input_size, hidden_size, bias=False)

        self.xh = nn.Linear(input_size, hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=True)

        self.layer_norm_rz_gate = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
        self.layer_norm_l_gate = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.dropout_prob = dropout_prob

        self.t1_hrz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.t1_layer_norm_rz_gate = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
        self.t1_hh = nn.Linear(hidden_size, hidden_size, bias=True)

        '''
        self.t2_hrz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.t2_layer_norm_rz_gate = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
        self.t2_hh = nn.Linear(hidden_size, hidden_size, bias=True)

        self.t3_hrz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.t3_layer_norm_rz_gate = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
        self.t3_hh = nn.Linear(hidden_size, hidden_size, bias=True)

        self.t4_hrz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.t4_layer_norm_rz_gate = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
        self.t4_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        '''

    '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
    '''
    def forward(self, x_t, h_tm1, x_m=None):

        assert ( x_t.dim() == 2 and h_tm1.dim() == 2 ), 'dim of hidden state should be 2'

        # 1-th layer LGRU
        x_rz_t, h_rz_tm1 = self.xrz(x_t), self.hrz(h_tm1)
        rz_t = tc.sigmoid( self.layer_norm_rz_gate( x_rz_t + h_rz_tm1 ) )
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]

        l_t = tc.sigmoid( self.layer_norm_l_gate( self.xl(x_t) + self.hl(h_tm1) ) )

        h_t_above = tc.tanh( self.xh(x_t) + r_t * self.hh(h_tm1) ) + l_t * self.wx(x_t)
        h_t_above = F.dropout(h_t_above, p=self.dropout_prob, training=self.training)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above

        # 1-th layer TGRU
        rz_t = self.t1_hrz(h_t)
        rz_t = tc.sigmoid( self.t1_layer_norm_rz_gate( rz_t ) )
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]
        h_t_above = tc.tanh( r_t * self.t1_hh(h_t) )
        h_t_above = F.dropout(h_t_above, p=self.dropout_prob, training=self.training)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above

        '''
        # 2-th layer TGRU
        rz_t = self.t2_hrz(h_t)
        rz_t = tc.sigmoid( self.t2_layer_norm_rz_gate( rz_t ) )
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]
        h_t_above = tc.tanh( r_t * self.t2_hh(h_t) )
        h_t_above = F.dropout(h_t_above, p=self.dropout_prob, training=self.training)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above

        # 3-th layer TGRU
        rz_t = self.t3_hrz(h_t)
        rz_t = tc.sigmoid( self.t3_layer_norm_rz_gate( rz_t ) )
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]
        h_t_above = tc.tanh( r_t * self.t3_hh(h_t) )
        h_t_above = F.dropout(h_t_above, p=self.dropout_prob, training=self.training)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above
        if x_m is not None:
            #h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
            h_t = x_m[:, None] * h_t

        # 4-th layer TGRU
        rz_t = self.t4_hrz(h_t)
        rz_t = tc.sigmoid( self.t4_layer_norm_rz_gate( rz_t ) )
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]
        h_t_above = tc.tanh( r_t * self.t4_hh(h_t) )
        h_t_above = F.dropout(h_t_above, p=self.dropout_prob, training=self.training)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above
        '''

        if x_m is not None:
            x_m = x_m[:, None]
            o_t = h_t * x_m
            h_t = o_t + h_tm1 * (1. - x_m)
        else: o_t = h_t

        return o_t, h_t


'''
Transition GRU with initial state as parameter:

    z_t = sigmoid( h_{t-1} dot U_hz + b_hz )
    r_t = sigmoid( h_{t-1} dot U_hr + b_xr )

    => zr_t = sigmoid( h_{t-1} dot U_hzr + b_hzr )
    slice ...

    h_above = tanh( r_t * ( h_{t-1} dot U_hh + b_hh ) )

    h_t = (1 - z_t) * h_{t-1} + z_t * h_above

all parameters are initialized in [-0.01, 0.01]
'''
class TGRU(nn.Module):

    def __init__(self, hidden_size, dropout_prob=None, prefix='TGRU', **kwargs):

        super(TGRU, self).__init__()

        self.hidden_size = hidden_size
        self.prefix = prefix

        self.hrz = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=True)

        self.layer_norm_rz_gate = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
        if dropout_prob is not None and 0. < dropout_prob <= 1.0:
            self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob

    '''
        x_m: mask of x_t
        h_tm1: previous state
    '''
    def forward(self, h_tm1, x_m=None):

        assert h_tm1.dim() == 2, 'dim of hidden state should be 2'

        rz_t = self.hrz(h_tm1)
        rz_t = tc.sigmoid( self.layer_norm_rz_gate( rz_t ) )
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]

        h_t_above = tc.tanh( r_t * self.hh(h_tm1) )

        if self.dropout_prob is not None and 0. < self.dropout_prob <= 1.0:
            h_t_above = self.dropout(h_t_above)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above

        if x_m is not None:
            #h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
            h_t = x_m[:, None] * h_t

        return h_t




