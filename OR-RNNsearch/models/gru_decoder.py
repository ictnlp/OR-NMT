import wargs
import torch as tc
import torch.nn as nn
from tools.utils import *
from models.nn_utils import Linear

'''
    Transition Gated Recurrent Unit network decoder
    input args:
        trg_emb:        class WordEmbedding
        enc_hid_size:   the size of TGRU hidden state in encoder
        dec_hid_size:   the size of TGRU hidden state in decoder
        n_layers:       layer nubmer of decoder
'''
class StackedGRUDecoder(nn.Module):

    def __init__(self, trg_emb, enc_hid_size=512, dec_hid_size=512, n_layers=2,
                 attention_type='multihead_additive', max_out=False,
                 rnn_dropout_prob=0.3, out_dropout_prob=0.5,
                 prefix='GRU_Decoder', **kwargs):

        super(StackedGRUDecoder, self).__init__()

        self.trg_word_emb = trg_emb
        n_embed = trg_emb.n_embed
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.s_init = Linear(2 * enc_hid_size, dec_hid_size, bias=True)
        self.gru_cell = nn.GRUCell(n_embed, dec_hid_size, bias=True)
        self.cgru_cell = nn.GRUCell(2 * enc_hid_size, dec_hid_size, bias=True)

        self.n_layers = n_layers
        if attention_type == 'additive':
            self.keys_transform = Linear(2 * enc_hid_size, dec_hid_size)
            from .attention import Additive_Attention
            self.attention = Additive_Attention(dec_hid_size, dec_hid_size)
        if attention_type == 'multihead_additive':
            self.keys_transform = Linear(2 * enc_hid_size, dec_hid_size, bias=False)
            from .attention import Multihead_Additive_Attention
            self.attention = Multihead_Additive_Attention(enc_hid_size, dec_hid_size)

        self.sigmoid = nn.Sigmoid()
        self.s_transform = Linear(dec_hid_size, dec_hid_size)
        self.y_transform = Linear(n_embed, dec_hid_size)
        self.c_transform = Linear(2 * enc_hid_size, dec_hid_size)
        self.src_transform = Linear(dec_hid_size, 2 * enc_hid_size)
        self.max_out = max_out
        self.out_dropout_prob = out_dropout_prob

    def init_state(self, annotations, xs_mask=None):

        assert annotations.dim() == 3  # (batch_size, max_L, dec_hid_size)
        uh = self.keys_transform(annotations)
        if xs_mask is not None:
            annotations = (annotations * xs_mask[:, :, None]).sum(1) / xs_mask.sum(1)[:, None]
        else:
            annotations = annotations.mean(1)

        return tc.tanh(self.s_init(annotations)), uh

    def sample_prev_y(self, k, ys_e, y_tm1_model=None, oracles=None, ss_prob=1.):

        batch_size = ys_e.size(0)
        if wargs.ss_type is not None and ss_prob < 1. and (wargs.greed_sampling or wargs.bleu_sampling):

            if wargs.greed_sampling is True:
                if oracles is not None:     # joint word and sentence level
                    with tc.no_grad():
                        _seed = tc.zeros(batch_size, 1, requires_grad=False).bernoulli_()
                        if wargs.gpu_id is not None: _seed = _seed.cuda()
                        y_tm1_oracle = y_tm1_model * _seed + oracles[k] * (1. - _seed)
                        #y_tm1_oracle = y_tm1_model.data.mul_(_seed) + y_tm1_oracle.data.mul_(1. - _seed)
                        #wlog('joint word and sent ... ')
                else:
                    y_tm1_oracle = y_tm1_model  # word-level oracle (w/o w/ noise)
                    #wlog('word level oracle ... ')
            else:
                y_tm1_oracle = oracles[k]   # sentence-level oracle
                #wlog('sent level oracle ... ')

            #uval = tc.rand(batch_size, 1)    # different word and differet batch
            #if wargs.gpu_id: uval = uval.cuda()
            # pick gold with the probability of ss_prob
            with tc.no_grad():
                _g = tc.bernoulli( ss_prob * tc.ones(batch_size, 1, requires_grad=False) )
                if wargs.gpu_id is not None: _g = _g.cuda()
                y_tm1 = ys_e[:, k, :] * _g + y_tm1_oracle * (1. - _g)
                #y_tm1 = schedule_sample_word(_h, _g, ss_prob, ys_e[k], y_tm1_oracle)
                #y_tm1 = ys_e[k].data.mul_(_g) + y_tm1_oracle.data.mul_(1. - _g)
        else:
            y_tm1 = ys_e[:, k, :]
            #g = self.sigmoid(self.w_gold(y_tm1) + self.w_hypo(y_tm1_oracle))
            #y_tm1 = g * y_tm1 + (1. - g) * y_tm1_oracle

        return y_tm1

    def step(self, s_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None):

        if not isinstance(y_tm1, tc.Tensor):
            if isinstance(y_tm1, int): y_tm1 = tc.tensor([y_tm1], dtype=tc.long, requires_grad=False)
            elif isinstance(y_tm1, list): y_tm1 = tc.tensor(y_tm1, dtype=tc.long, requires_grad=False)
            if wargs.gpu_id is not None: y_tm1 = y_tm1.cuda()
            _, y_tm1 = self.trg_word_emb(y_tm1)

        state = self.gru_cell(y_tm1, s_tm1)
        if y_mask is not None: state = state * y_mask[:, None]
        # state: (batch_size, d_dec_hid)

        alpha, context = self.attention(state, xs_h, uh, xs_mask)
        # alpha:   [batch_size, n_head, key_len] or [batch_size, key_len]
        # context: [batch_size, 2 * enc_hid_size]
        if y_mask is not None: context = context * y_mask[:, None]

        s_t = self.cgru_cell(context, state)
        if y_mask is not None: s_t = s_t * y_mask[:, None]

        return context, s_t, y_tm1, alpha

    def forward(self, xs_h, ys, xs_mask, ys_mask, ss_prob=1., oracles=None):

        s_tm1, uh = self.init_state(xs_h, xs_mask)
        batch_size, y_Lm1 = ys.size(0), ys.size(1)

        if ys.dim() == 3: ys_e = ys
        else: _, ys_e = self.trg_word_emb(ys)   # (batch_size, y_Lm1, d_trg_emb)

        logits, attends, contexts, y_tm1_model = [], [], [], ys_e[:, 0, :]
        for k in range(y_Lm1):

            y_tm1 = self.sample_prev_y(k, ys_e, y_tm1_model, oracles, ss_prob)
            context, s_tm1, _, alpha_ij = self.step(s_tm1, xs_h, uh, y_tm1, xs_mask, ys_mask[:, k])
            logit = self.step_out(y_tm1, context, s_tm1)
            logits.append(logit)
            attends.append(alpha_ij)
            contexts.append(context)

            if wargs.ss_type is not None and ss_prob < 1. and wargs.greed_sampling is True:
                logit = self.classifier.pred_map(logit, noise=wargs.greed_gumbel_noise)
                y_tm1_model = logit.max(-1)[1]
                _, y_tm1_model = self.trg_word_emb(y_tm1_model)
                #wlog('word-level greedy sampling, noise {}, ss_prob {}'.format(
                    #wargs.greed_gumbel_noise, ss_prob))

        logits = tc.stack(logits, dim=1) * ys_mask[:, :, None]    # (batch_size, y_Lm1, d_dec_hid)
        attends = tc.stack(attends, dim=1) * ys_mask[:, :, None]  # (batch_size, y_Lm1, key_len)
        contexts = tc.stack(contexts, dim=1) * ys_mask[:, :, None]# (batch_size, y_Lm1, 2 * enc_hid_size)

        return {
            'logit': logits,
            'attend': attends,
            'context': contexts
        }

    def step_out(self, y, c, s):

        # (batch_size, y_Lm1, dec_hid_size)
        logit = self.y_transform(y) + self.c_transform(c) + self.s_transform(s)
        logit = self.src_transform(logit)

        if self.max_out is True:
            if logit.dim() == 2:    # for decoding
                logit = logit.view(logit.size(0), logit.size(1)/2, 2)
            elif logit.dim() == 3:
                logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)
            logit = logit.max(-1)[0]

        logit = tc.tanh(logit)
        logit = F.dropout(logit, p=self.out_dropout_prob, training=self.training)

        return logit


