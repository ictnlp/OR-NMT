from __future__ import division, print_function

import math
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import MaskSoftmax, Linear
np.set_printoptions(threshold=np.nan)

class Additive_Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Additive_Attention, self).__init__()
        self.sa = nn.Linear(dec_hid_size, align_size)
        self.tanh = nn.Tanh()
        self.maskSoftmax = MaskSoftmax()
        self.a1 = nn.Linear(align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        e_ij = self.a1( self.tanh(self.sa(s_tm1)[:, None, :] + uh) ).squeeze(-1)

        e_ij = self.maskSoftmax(e_ij, mask=xs_mask, dim=1)  # (batch_size, key_len)
        # weighted sum of the h_j: (batch_size, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(1)

        return e_ij, attend

class Multihead_Additive_Attention(nn.Module):

    #dec_hid_size:   the dimension of n_head keys/values/queries: dec_hid_size % n_head == 0
    #n_head:    number of parallel heads.
    def __init__(self, enc_hid_size, dec_hid_size, n_head=8):

        super(Multihead_Additive_Attention, self).__init__()

        assert dec_hid_size % n_head == 0, 'dec_hid_size {} divided by n_head {}.'.format(dec_hid_size, n_head)
        self.n_head = n_head
        self.linear_query = Linear(dec_hid_size, dec_hid_size, bias=False)
        #self.mSoftMax = MaskSoftmax()
        dim_per_head = dec_hid_size // n_head
        self.a1 = Linear(dim_per_head, 1, bias=False)
        self.final_proj = Linear(2 * enc_hid_size, 2 * enc_hid_size, bias=True)

    '''
        Compute the context vector and the attention vectors.
        Args:
           q (FloatTensor): query [batch_size, dec_hid_size]             ->  hidden state
           v (FloatTensor): value [batch_size, key_len, 2*dec_hid_size]  ->  annotations
           k (FloatTensor): key [batch_size, key_len, dec_hid_size]      ->  uh
           attn_mask: binary mask indicating
                    which keys have non-zero attention [batch_size, key_len]
        Returns:
           (FloatTensor, FloatTensor) :
           * context vectors [batch_size, 2 * dec_hid_size]
           * probability            [batch_size, n_head, key_len]
    '''
    def forward(self, q, v, k, attn_mask=None):

        def split_heads(x, nhead):
            return x.view(x.size(0), x.size(1), nhead, x.size(-1) // nhead).permute(0, 2, 1, 3)

        def combine_heads(x, nhead):
            return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), nhead * x.size(-1))

        q = self.linear_query(q)
        # 1. project key, value, and query
        q = split_heads(q[:, None, :], self.n_head) # [batch_size, n_head, 1, dim_per_head]
        k = split_heads(k, self.n_head)             # [batch_size, n_head, key_len, dim_per_head]

        hidden = tc.tanh(q + k)
        attn = self.a1(hidden).squeeze(-1)          # [batch_size, n_head, key_len]
        if attn_mask is not None:   # [batch_size, key_len]
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn).byte()    # expand along n_head dim
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill_(1. - attn_mask, float('-inf'))

        # 3. apply attention dropout and compute context vectors
        #alpha = self.mSoftMax(attn)            # [batch_size, n_head, key_len]
        alpha = F.softmax(attn, dim=-1)         # [batch_size, n_head, key_len]

        v = split_heads(v, self.n_head)             # [batch_size, n_head, key_len, 2*dim_per_head]
        attn = alpha[:, :, :, None] * v             # [batch_size, n_head, key_len, 2*dim_per_head]
        attn = combine_heads(attn, self.n_head)     # [batch_size, key_len, 2*d_model]

        attn = self.final_proj(attn.sum(1))       # [batch_size, 2 * d_model]

        alpha = alpha.sum(1) / self.n_head # get the attention of the first head, [batch_size, key_len]
        #alpha = alpha[:, 0, :].transpose(0, 1)  # get the attention of the first head, [key_len, batch_size]

        return alpha, attn

class MultiHeadAttention(nn.Module):

    #d_model(int):   the dimension of n_head keys/values/queries: d_model % n_head == 0
    #n_head(int):    number of parallel heads.
    def __init__(self, d_model, n_head, dropout_prob=0.1):

        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0, 'd_model {} divided by n_head {}.'.format(d_model, n_head)
        self.dim_per_head = d_model // n_head
        self.scaling = self.dim_per_head ** -0.5
        self.d_model = d_model
        self.n_head = n_head

        self.kqv_proj_weight = nn.Parameter(tc.Tensor(3 * d_model, d_model))
        self.kqv_proj_bias = nn.Parameter(tc.Tensor(3 * d_model))
        #self.linear_keys = nn.Linear(d_model, n_head * self.dim_per_head)
        #self.linear_values = nn.Linear(d_model, n_head * self.dim_per_head)
        #self.linear_query = nn.Linear(d_model, n_head * self.dim_per_head)
        #self.mSoftMax = MaskSoftmax()
        self.dropout_prob = dropout_prob
        #self.final_proj = nn.Linear(d_model, d_model)
        self.final_proj_weight = nn.Parameter(tc.Tensor(d_model, d_model))
        self.final_proj_bias = nn.Parameter(tc.Tensor(d_model))

        nn.init.xavier_uniform_(self.kqv_proj_weight)
        nn.init.xavier_uniform_(self.final_proj_weight)
        nn.init.constant_(self.kqv_proj_bias, 0.)
        nn.init.constant_(self.final_proj_bias, 0.)

    '''
        Compute the context vector and the attention vectors.
        Args:
           k (FloatTensor): key vectors [batch_size, key_len, d_model]
           v (FloatTensor): value vectors [batch_size, key_len, d_model]
           q (FloatTensor): query vectors  [batch_size, query_len, d_model]
           attn_mask: binary mask indicating
                    which keys have non-zero attention [batch_size, query_len, key_len]
        Returns:
           (FloatTensor, FloatTensor, FloatTensor) :
           * context vectors [batch_size, query_len, d_model]
           * all attention vectors [batch_size, n_head, query_len, key_len]
           * one of the attention vectors [batch_size, query_len, key_len]
    '''
    def forward(self, k, v, q, attn_mask=None, query_mask=None):

        batch_size, n_head = k.size(0), self.n_head

        def split_heads(x):
            return x.view(batch_size, -1, n_head, self.dim_per_head).transpose(1, 2)

        def combine_heads(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, n_head * self.dim_per_head)

        # 1. project key, value, and query
        #k = split_heads(self.linear_keys(k)) # [batch_size, n_head, key_len, dim_per_head]
        #v = split_heads(self.linear_values(v)) # [batch_size, n_head, key_len, dim_per_head]
        #q = split_heads(self.linear_query(q))  # [batch_size, n_head, query_len, dim_per_head]
        k = F.linear(k, self.kqv_proj_weight[0 : self.d_model, :],
                     self.kqv_proj_bias[0 : self.d_model])
        q = F.linear(q, self.kqv_proj_weight[self.d_model : 2 * self.d_model, :],
                     self.kqv_proj_bias[self.d_model : 2 * self.d_model])
        v = F.linear(v, self.kqv_proj_weight[2 * self.d_model :, :],
                     self.kqv_proj_bias[2 * self.d_model :])
        k = split_heads(k)
        q = split_heads(q)
        v = split_heads(v)

        # 2. calculate and scale scores: Attention(Q,K,V) = softmax(QK/sqrt(d_k))*V
        q = q * self.scaling # [batch_size, n_head, query_len, dim_per_head]
        attn = tc.matmul(q, k.transpose(2, 3)) #[batch_size, n_head, query_len, key_len]

        if attn_mask is not None:   # [batch_size, query_len, key_len]
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn).byte()    # expand along n_head dim
            #print('-------------------')
            #print(attn_mask.size())
            #print(attn_mask.cpu().numpy())
            #print(attn.size())
            #print(attn.detach().cpu().numpy())
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            #attn.masked_fill_(1 - attn_mask, float('-inf'))
            #attn.masked_fill_(1, float('-inf'))
            attn = attn.masked_fill(attn_mask, -1e18)
            #print(attn.size())
            #print(attn.detach().cpu().numpy())

        # 3. apply attention dropout and compute context vectors
        #attn = self.mSoftMax(attn, dim=-1)
        attn = F.softmax(attn, dim=-1)
        #print('softmax.....................')
        #print(attn.size())
        #print(attn.detach().cpu().numpy())
        attn = F.dropout(attn, p=self.dropout_prob, training=self.training) # [batch_size, n_head, query_len, key_len]
        context = tc.matmul(attn, v)    # [batch_size, n_head, query_len, dim_per_head]
        #context = tc.bmm(attn.contiguous().view(-1, query_len, key_len),
        #                 v.contiguous().view(-1, key_len, dim_per_head))    # [batch_size, n_head, query_len, dim_per_head]
        #context = context.view(batch_size, n_head, query_len, dim_per_head)
        context = combine_heads(context)             # [batch_size, query_len, n_head * dim_per_head]

        #context = self.final_proj(context)   # [batch_size, query_len, d_model]
        context = F.linear(context, self.final_proj_weight, self.final_proj_bias)   # [batch_size, query_len, d_model]

        attn = attn.sum(dim=1) / self.n_head    # average attention weights over heads
        if query_mask is not None:
            query_mask = query_mask[:, :, None]
            context = context * query_mask
            attn = attn * query_mask

        return context, attn

import torch
class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, '{}, {}'.format(key_padding_mask.size(0), bsz)
            assert key_padding_mask.size(1) == src_len, '{}, {}'.format(key_padding_mask.size(1), src_len)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

