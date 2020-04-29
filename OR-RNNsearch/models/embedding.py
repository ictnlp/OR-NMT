import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import PAD, MAX_SEQ_SIZE, wlog
import wargs

def make_positions(tensor, padding_idx, left_pad, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    if onnx_trace:
        range_buf = torch._dim_arange(like=tensor, dim=1) + padding_idx + 1
        mask = tensor.ne(padding_idx)
        positions = range_buf.expand_as(tensor)
        if left_pad:
            positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
        return positions * mask.long() + padding_idx * (1 - mask.long())

    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])

'''
def make_positions(tensor, padding_idx):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        tc.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    return tensor.clone().masked_scatter_(mask, positions[mask])
'''

def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m

import torch
class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]."""
        #bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = input.size(0), input.size(1)
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = (timestep.int() + 1).long() if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights[self.padding_idx + pos, :].unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx, self.left_pad, self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

'''
Implements the sinusoidal positional encoding for non-recurrent neural networks
Args:
   dropout (float): dropout parameter
   n_embed (int): embedding size
'''
class PositionalEncoding(nn.Module):

    def __init__(self, dropout_prob, n_embed, max_len=MAX_SEQ_SIZE):

        pe = tc.zeros(max_len, n_embed)
        position = tc.arange(0, max_len).unsqueeze(1)
        div_term = tc.exp((tc.arange(0, n_embed, 2) * -(math.log(10000.0) / n_embed)).float())
        inter_term = position.float() * div_term
        # keep dim 0 for padding token position encoding zero vector
        pe[1:, 0::2] = tc.sin(inter_term)[1:]
        pe[1:, 1::2] = tc.cos(inter_term)[1:]
        # [5000, 1] * [256] = [5000, 256] 
        #pe[:, 0::2] = tc.sin(position.float() * div_term)
        #pe[:, 1::2] = tc.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)    # [5000, 512] -> [5000, 1, 512]
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.n_embed = n_embed
        wlog('pe: {}'.format(pe.size()))

        self.dropout_prob =  dropout_prob
        if dropout_prob is not None and 0. < dropout_prob <= 1.0:
            wlog('with emb dropout prob = {} ...'.format(dropout_prob))
            self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, emb):

        emb = emb * math.sqrt(self.n_embed)
        emb = emb + self.pe[:emb.size(0)]
        if self.dropout_prob is not None and 0. < self.dropout_prob < 1.0: emb = self.dropout(emb)

        return emb

class WordEmbedding(nn.Module):

    def __init__(self,
                 n_vocab,
                 n_embed=512,
                 emb_dropout=0.,
                 position_encoding=False,
                 prefix='WordEmbedding'):

        super(WordEmbedding, self).__init__()
        wlog('WordEmbedding_{}'.format(prefix))
        self.position_encoding = position_encoding
        self.we = nn.Embedding(n_vocab, n_embed, padding_idx=PAD)
        nn.init.normal_(self.we.weight, mean=0, std=n_embed ** -0.5)
        nn.init.constant_(self.we.weight[PAD], 0)
        self.n_embed = n_embed
        if position_encoding is True:
            wlog('with position emb ...')
            #self.pe = PositionalEncoding(emb_dropout, n_embed)
            #self.spe = PositionalEmbedding(MAX_SEQ_SIZE, n_embed, PAD, left_pad=False, learned=False)
        wlog('with emb dropout prob = {} ...'.format(emb_dropout))
        self.emb_dropout = emb_dropout

    def add_timing_signal(self, x_emb, min_timescale=1.0, max_timescale=1.0e4, name=None):

        length, channels = x_emb.size(1), x_emb.size(2)
        position = tc.arange(length).float()
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tc.exp(
            (tc.arange(num_timescales).float()) * -log_timescale_increment
        )

        scaled_time = position[:, None] * inv_timescales[None, :]
        signal = tc.cat([tc.sin(scaled_time), tc.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, channels % 2, 0, 0))
        signal = signal.reshape(1, length, channels)
        if wargs.gpu_id is not None: signal = signal.cuda()

        return x_emb + signal * (float(channels) ** -0.5)

    def forward(self, x):

        x_w_emb = self.we(x)
        if self.position_encoding is True:
            x_wp_emb = self.add_timing_signal(x_w_emb)
            #scale = math.sqrt(self.n_embed)
            #x_wp_emb = scale * x_w_emb + self.spe(x)
        else:
            x_wp_emb = x_w_emb

        x_wp_emb = F.dropout(x_wp_emb, p=self.emb_dropout, training=self.training)

        return x_w_emb, x_wp_emb






















