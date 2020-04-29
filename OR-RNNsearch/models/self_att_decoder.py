from __future__ import division, print_function

import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import MAX_SEQ_SIZE, wlog, PAD
from .nn_utils import PositionwiseFeedForward
from .attention import MultiHeadAttention
#from .attention import MultiheadAttention
np.set_printoptions(threshold=np.nan)

'''
Get an attention mask to avoid using the subsequent info.
Args: d_model: int
Returns: (LongTensor): future_mask [1, d_model, d_model]
'''
def get_attn_future_mask(size):

    attn_shape = (1, size, size)
    future_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    future_mask = tc.from_numpy(future_mask)

    return future_mask

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

'''
Compose with three layers
    Args:
        d_model(int): the dimension of keys/values/queries in
                      MultiHeadAttention, also the input size of
                      the first-layer of the PositionwiseFeedForward.
        n_head(int): the number of head for MultiHeadAttention.
        hidden_size(int): the second-layer of the PositionwiseFeedForward.
        droput(float): dropout probability(0-1.0).
'''
class SelfAttDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 n_head=8,
                 d_ff_filter=2048,
                 att_dropout=0.3,
                 residual_dropout=0.,
                 relu_dropout=0.,
                 self_attn_type='scaled-dot',
                 decoder_normalize_before=False):

        super(SelfAttDecoderLayer, self).__init__()

        self.decoder_normalize_before = decoder_normalize_before
        self.layer_norm_0 = nn.LayerNorm(d_model, elementwise_affine=True)

        self.self_attn_type = self_attn_type
        if self_attn_type == 'scaled-dot':
            self.self_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)
            #self.self_attn = MultiheadAttention(d_model, n_head, dropout=att_dropout)
        elif self_attn_type == 'average':
            self.self_attn = AverageAttention(d_model, dropout_prob=att_dropout)

        self.residual_dropout_prob = residual_dropout

        self.layer_norm_1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.trg_src_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)
        #self.trg_src_attn = MultiheadAttention(d_model, n_head, dropout=att_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff_filter, d_model, dropout_prob=relu_dropout)
        mask = get_attn_future_mask(MAX_SEQ_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, x, enc_output, trg_self_attn_mask=None, trg_src_attn_mask=None, query_mask=None):
    #def forward(self, x, encoder_out, encoder_padding_mask, incremental_state=None,
    #            self_attn_mask=None, self_attn_padding_mask=None):
        '''
        Args:
            x (FloatTensor):                [batch_size, trg_len, d_model]
            enc_output (FloatTensor):       [batch_size, src_len, d_model]
            trg_self_attn_mask (LongTensor):[batch_size, trg_len, trg_len]
            trg_src_attn_mask  (LongTensor):[batch_size, trg_len, src_len]
        Returns: (FloatTensor, FloatTensor, FloatTensor, FloatTensor):
            dec_output:         [batch_size, trg_len, d_model]
            trg_self_attns:     [batch_size, n_head, trg_len, trg_len]
            trg_src_attns:      [batch_size, n_head, trg_len, src_len]
            one_dec_enc_attn:   [batch_size, trg_len, src_len]
        '''
        #if query_mask is not None:
        trg_self_attn_mask = tc.gt(trg_self_attn_mask +
                         self.mask[:, :trg_self_attn_mask.size(-1),
                                   :trg_self_attn_mask.size(-1)], 0)
        # target self-attention
        residual = x
        if self.decoder_normalize_before is True:
            x = self.layer_norm_0(x)     # before 'n' for preprocess

        # trg_self_attn_mask: (batch_size, trg_len, trg_len)
        if self.self_attn_type == 'scaled-dot':
            x, trg_self_attns = self.self_attn(x, x, x, attn_mask=trg_self_attn_mask,
                                               query_mask=query_mask)
            '''
            x, trg_self_attns = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=None,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            '''
            # query:                [batch_size, trg_len, d_model]
            # trg_self_attns:       [batch_size, n_head, trg_len, trg_len]
            # one_dec_self_attn:    [batch_size, trg_len, trg_len]
        elif self.self_attn_type == 'average':
            query, attn = self.self_attn(input_norm, mask=trg_self_attn_mask,
                                         layer_cache=layer_cache, step=step)

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x    # 'da' for postprocess
        if self.decoder_normalize_before is False:
            x = self.layer_norm_0(x)

        # encoder-decoder attention
        residual = x
        if self.decoder_normalize_before is True:
            x = self.layer_norm_1(x)   # before 'n' for preprocess

        # trg_src_attn_mask: (batch_size, trg_len, src_len)
        x, trg_src_attns = self.trg_src_attn(enc_output, enc_output, x, attn_mask=trg_src_attn_mask, query_mask=query_mask)
        '''
        x, trg_src_attns = self.trg_src_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=(not self.training and True),
        )
        '''
        # x:                    [batch_size, trg_len, d_model]
        # trg_src_attns:        [batch_size, trg_len, src_len]

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x    # before 'da' for postprocess
        if self.decoder_normalize_before is False:
            x = self.layer_norm_1(x)

        # feed forward
        residual = x
        if self.decoder_normalize_before is True:
            x = self.layer_norm_2(x)   # 'n' for preprocess

        x = self.pos_ffn(x)

        x = F.dropout(x, p=self.residual_dropout_prob, training=self.training)
        x = residual + x    # 'da' for postprocess
        if self.decoder_normalize_before is False:
            x = self.layer_norm_2(x)

        return x, trg_self_attns, trg_src_attns

''' A decoder model with self attention mechanism '''
class SelfAttDecoder(nn.Module):

    def __init__(self, trg_emb,
                 n_layers=6,
                 d_model=512,
                 n_head=8,
                 d_ff_filter=1024,
                 att_dropout=0.3,
                 residual_dropout=0.,
                 relu_dropout=0.,
                 self_attn_type='scaled-dot',
                 proj_share_weight=False,
                 decoder_normalize_before=False):

        wlog('Transformer decoder ========================= ')
        wlog('\ttrg_word_emb:       {}'.format(trg_emb.we.weight.size()))
        wlog('\tn_layers:           {}'.format(n_layers))
        wlog('\tn_head:             {}'.format(n_head))
        wlog('\td_model:            {}'.format(d_model))
        wlog('\td_ffn_filter:       {}'.format(d_ff_filter))
        wlog('\tatt_dropout:        {}'.format(att_dropout))
        wlog('\tresidual_dropout:   {}'.format(residual_dropout))
        wlog('\trelu_dropout:       {}'.format(relu_dropout))
        wlog('\tproj_share_weight:  {}'.format(proj_share_weight))

        super(SelfAttDecoder, self).__init__()

        self.layer_stack = nn.ModuleList([
            SelfAttDecoderLayer(d_model,
                                n_head,
                                d_ff_filter,
                                att_dropout=att_dropout,
                                residual_dropout=residual_dropout,
                                relu_dropout=relu_dropout,
                                self_attn_type=self_attn_type,
                                decoder_normalize_before=decoder_normalize_before)
            for _ in range(n_layers)])

        self.trg_word_emb = trg_emb
        if decoder_normalize_before is True:
            self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.decoder_normalize_before = decoder_normalize_before

    def forward(self, trg_seq, src_seq, enc_output, trg_mask=None, src_mask=None):

        src_B, src_L = src_seq.size()
        trg_B, trg_L = trg_seq.size()
        assert src_B == trg_B

        '''
        Get an attention mask to avoid using the subsequent info.
        array([[[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]]], dtype=uint8)
        '''
        #trg_self_attn_mask = None if trg_mask is None else (1-trg_mask).byte().unsqueeze(1).expand(trg_B, trg_L, trg_L)
        #trg_src_attn_mask = None if src_mask is None else (1-src_mask).byte().unsqueeze(1).expand(src_B, trg_L, src_L)
        trg_self_attn_mask = trg_seq.data.eq(PAD).byte().unsqueeze(1).expand(trg_B, trg_L, trg_L)  # [B, 1, T_tgt]
        trg_src_attn_mask = src_seq.data.eq(PAD).byte().unsqueeze(1).expand(src_B, trg_L, src_L)  # [B, 1, T_src]
        '''
        with tc.no_grad():
            if trg_mask is not None:
                future_mask = tc.tril(tc.ones(trg_L, trg_L), diagonal=0, out=None).cuda()
                trg_self_attn_mask = tc.gt(trg_self_attn_mask + future_mask[None, :, :], 1)
        '''
        _, x = self.trg_word_emb(trg_seq)

        #nlayer_outputs, nlayer_self_attns, nlayer_attns = [], [], []
        #incremental_state = None
        #x = x.transpose(0, 1)
        #if src_mask is not None: src_mask = 1 - src_mask.byte()
        for dec_layer in self.layer_stack:
            x, trg_self_attns, trg_src_attns = dec_layer(
                x, enc_output,
                trg_self_attn_mask=trg_self_attn_mask,
                trg_src_attn_mask=trg_src_attn_mask,
                query_mask=trg_mask)
            #x, trg_self_attns, trg_src_attns = dec_layer(
            #    x, enc_output, src_mask, incremental_state,
            #    self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            #)
            #nlayer_outputs += [dec_output]
            #nlayer_self_attns += [trg_self_attns]
            #nlayer_attns += [trg_src_attns]
        #x = x.transpose(0, 1)

        if self.decoder_normalize_before is True:
            x = self.layer_norm(x)    # layer norm for the last layer output

        #return (dec_output, nlayer_self_attns, nlayer_attns)
        return x, trg_self_attns, trg_src_attns

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = tc.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = tc.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


