from __future__ import division
import torch as tc
import torch.nn as nn
import wargs
from gru import TransiLNCell
from tools.utils import *

'''
    Bi-directional Transition Gated Recurrent Unit network encoder
    input args:
        src_emb:        class WordEmbedding
        enc_hid_size:   the size of TGRU hidden state
'''
class StackedTransEncoder(nn.Module):

    def __init__(self,
                 src_emb,
                 enc_hid_size=512,
                 rnn_dropout=0.3,
                 prefix='Encoder', **kwargs):

        super(StackedTransEncoder, self).__init__()

        self.src_word_emb = src_emb
        n_embed = src_emb.n_embed
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.forw_cell = TransiLNCell(input_size=n_embed,
                                      hidden_size=enc_hid_size,
                                      dropout_prob=rnn_dropout,
                                      prefix=f('Forw_TransiLNCell'))
        self.back_cell = TransiLNCell(input_size=n_embed,
                                      hidden_size=enc_hid_size,
                                      dropout_prob=rnn_dropout,
                                      prefix=f('Back_TransiLNCell'))

        self.enc_hid_size = enc_hid_size

    def forward(self, xs, xs_mask=None, h0=None):

        batch_size, max_L = xs.size(0), xs.size(1)
        if xs.dim() == 3: xs_e = xs
        else: x_w_e, xs_e = self.src_word_emb(xs)
        # (batch_size, n_embed)

        f_anns, b_anns = [], []
        f_h = b_h = h0 if h0 else tc.zeros(batch_size, self.enc_hid_size, requires_grad=False)
        if wargs.gpu_id is not None: f_h, b_h = f_h.cuda(), b_h.cuda()
        for f_idx in range(max_L):
            b_idx = max_L - f_idx - 1
            f_inp, b_inp = xs_e[:, f_idx, :], xs_e[:, b_idx, :]
            f_mask = xs_mask[:, f_idx] if xs_mask is not None else None
            b_mask = xs_mask[:, b_idx] if xs_mask is not None else None
            f_out, f_h = self.forw_cell(f_inp, f_h, f_mask)
            b_out, b_h = self.back_cell(b_inp, b_h, b_mask)
            f_anns.append(f_out)
            b_anns.append(b_out)
        f_anns, b_anns = tc.stack(f_anns, dim=1), tc.stack(b_anns[::-1], dim=1)

        anns = tc.cat([f_anns, b_anns], dim=-1)   # (batch_size, max_L, 2*enc_hid_size)

        return anns * xs_mask[:, :, None]

