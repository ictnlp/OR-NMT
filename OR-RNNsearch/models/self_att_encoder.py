from __future__ import division, print_function

import torch.nn as nn
from tools.utils import *
import torch.nn.functional as F
from .nn_utils import PositionwiseFeedForward
from .attention import MultiHeadAttention
#from .attention import MultiheadAttention

'''
    Args:
        d_model(int): the dimension of keys/values/queries in
                      MultiHeadAttention, also the input size of
                      the first-layer of the PositionwiseFeedForward.
        n_head(int): the number of head for MultiHeadAttention.
        hidden_size(int): the second-layer of the PositionwiseFeedForward.
        droput(float): dropout probability(0-1.0).
'''
class SelfAttEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 n_head=8,
                 d_ff_filter=2048,
                 att_dropout=0.3,
                 residual_dropout=0.0,
                 relu_dropout=0.0,
                 encoder_normalize_before=False):

        super(SelfAttEncoderLayer, self).__init__()
        self.layer_norm_0 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)
        #self.self_attn = MultiheadAttention(d_model, n_head, dropout=att_dropout)
        self.residual_dropout_prob = residual_dropout
        self.layer_norm_1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff_filter, d_model, dropout_prob=relu_dropout)
        self.encoder_normalize_before = encoder_normalize_before

    def forward(self, x, self_attn_mask=None, query_mask=None):
    #def forward(self, x, encoder_padding_mask=None):
        # x (FloatTensor):         [batch_size, src_L, d_model]
        # self_attn_mask(LongTensor):   [batch_size, src_L, src_L]
        # return:                       [batch_size, src_L, d_model]

        # self attention
        residual = x
        if self.encoder_normalize_before is True:
            x = self.layer_norm_0(x)   # before 'n' for source self attention preprocess
        x, enc_self_attns = self.self_attn(x, x, x, attn_mask=self_attn_mask, query_mask=query_mask)
        #x, enc_self_attns = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        # enc_output: (B_q, L_q, d_model), enc_self_attns: (B_q, L_q, L_k)

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x     # 'da' for self attention postprocess
        if self.encoder_normalize_before is False:
            x = self.layer_norm_0(x)   # after 'n' for source self attention preprocess

        residual = x
        # feed forward
        if self.encoder_normalize_before is True:
            x = self.layer_norm_1(x)        # before 'n' for feedforward preprocess

        x = self.pos_ffn(x)

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x   # 'da' for feedforward postprocess

        if self.encoder_normalize_before is False:
            x = self.layer_norm_1(x)        # after 'n' for feedforward preprocess

        # x:                    [batch_size, src_L, d_model]
        # enc_self_attns:       [batch_size, src_L, src_L]
        # one_enc_self_attn:    [batch_size, src_L, src_L]
        return x, enc_self_attns

''' A encoder model with self attention mechanism '''
class SelfAttEncoder(nn.Module):

    def __init__(self,
                 src_emb,
                 n_layers=6,
                 d_model=512,
                 n_head=8,
                 d_ff_filter=2048,
                 att_dropout=0.3,
                 residual_dropout=0.0,
                 relu_dropout=0.0,
                 encoder_normalize_before=False):

        super(SelfAttEncoder, self).__init__()

        wlog('Transformer encoder ========================= ')
        wlog('\tsrc_word_emb:       {}'.format(src_emb.we.weight.size()))
        wlog('\tn_layers:           {}'.format(n_layers))
        wlog('\tn_head:             {}'.format(n_head))
        wlog('\td_model:            {}'.format(d_model))
        wlog('\td_ffn_filter:       {}'.format(d_ff_filter))
        wlog('\tatt_dropout:        {}'.format(att_dropout))
        wlog('\tresidual_dropout:   {}'.format(residual_dropout))
        wlog('\trelu_dropout:       {}'.format(relu_dropout))

        self.embed = src_emb

        self.layer_stack = nn.ModuleList([
            SelfAttEncoderLayer(d_model,
                                n_head,
                                d_ff_filter,
                                att_dropout=att_dropout,
                                residual_dropout=residual_dropout,
                                relu_dropout=relu_dropout,
                                encoder_normalize_before=encoder_normalize_before)
            for _ in range(n_layers)])

        if encoder_normalize_before is True:
            self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.encoder_normalize_before = encoder_normalize_before

    def forward(self, src_seq, src_mask=None):

        batch_size, src_L = src_seq.size()
        # word embedding look up
        _, x = self.embed(src_seq)
        #nlayer_outputs, nlayer_attns = [], []
        #src_self_attn_mask = None if src_mask is None else (1-src_mask).byte().unsqueeze(1).expand(batch_size, src_L, src_L)
        src_self_attn_mask = src_seq.data.eq(PAD).byte().unsqueeze(1).expand(batch_size, src_L, src_L)  # [B, 1, T_tgt]
        for enc_layer in self.layer_stack:
            # enc_output: (B_q, L_q, d_model), enc_self_attns: (B, L_q, L_k)
            x, enc_self_attns = enc_layer(x, src_self_attn_mask, query_mask=src_mask)
            #x, enc_self_attns = enc_layer(x, src_mask)
            #nlayer_outputs += [enc_output]
            #nlayer_attns += [enc_self_attns]
        #x = x.transpose(0, 1)

        if self.encoder_normalize_before is True:
            x = self.layer_norm(x)    # layer norm for the last layer output

        # nlayer_outputs:   n_layers: [ [batch_size, src_L, d_model], ... ]
        # nlayer_attns:     n_layers: [ [batch_size, src_L, src_L], ... ]
        # one_enc_self_attn:          [batch_size, src_L, src_L]
        #return (enc_output, nlayer_attns)
        return x, enc_self_attns


