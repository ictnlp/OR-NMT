from __future__ import division

import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import wargs
from tools.utils import wlog, PAD, schedule_bow_lambda
from models.nn_utils import MaskSoftmax, MyLogSoftmax, Linear

epsilon = 1e-20
class Classifier(nn.Module):

    def __init__(self, input_size, output_size, trg_word_emb=None, loss_norm='tokens',
                 label_smoothing=0., emb_loss=False, bow_loss=False):

        super(Classifier, self).__init__()
        if emb_loss is True:
            assert trg_word_emb is not None, 'embedding loss needs target embedding'
            self.trg_word_emb = trg_word_emb.we
            self.euclidean_dist = nn.PairwiseDistance(p=2, eps=1e-06, keepdim=True)
        self.emb_loss = emb_loss
        if bow_loss is True:
            wlog('using the bag of words loss')
            self.sigmoid = nn.Sigmoid()
            #self.ctx_map_vocab = Linear(2 * input_size, output_size, bias=True)
            #self.softmax = MaskSoftmax()
        self.bow_loss = bow_loss

        self.map_vocab = Linear(input_size, output_size, bias=True)
        nn.init.normal_(self.map_vocab.weight, mean=0, std=input_size ** -0.5)
        if wargs.proj_share_weight is True:
            assert input_size == wargs.d_trg_emb
            wlog('copying weights of target word embedding into classifier')
            self.map_vocab.weight = trg_word_emb.we.weight
        self.log_prob = MyLogSoftmax(wargs.self_norm_alpha)

        assert 0. <= label_smoothing <= 1., 'label smoothing value should be in [0, 1]'
        wlog('NLL loss with label_smoothing: {}'.format(label_smoothing))
        if label_smoothing == 0. or self.bow_loss is True:
            weight = tc.ones(output_size, requires_grad=False)
            weight[PAD] = 0   # do not predict padding, same with ingore_index
            self.criterion = nn.NLLLoss(weight, ignore_index=PAD, reduction='sum')
            #self.criterion = nn.NLLLoss(weight, ignore_index=PAD, size_average=False)
        if 0. < label_smoothing <= 1.:
            # all non-true labels are uniformly set to low-confidence
            self.smoothing_value = label_smoothing / (output_size - 2)
            one_hot = tc.full((output_size, ), self.smoothing_value)
            one_hot[PAD] = 0.
            self.register_buffer('one_hot', one_hot.unsqueeze(0))
            self.confidence = 1.0 - label_smoothing

        self.output_size = output_size
        self.softmax = MaskSoftmax()
        self.loss_norm = loss_norm
        self.label_smoothing = label_smoothing

    def pred_map(self, logit, noise=None):

        logit = self.map_vocab(logit)

        if noise is not None:
            with tc.no_grad():
                logit.data.add_( -tc.log(-tc.log(tc.Tensor(
                    logit.size()).cuda().uniform_(0, 1) + epsilon) + epsilon) ) / noise

        return logit

    def logit_to_prob(self, logit, gumbel=None, tao=None):

        # (L, B)
        d1, d2, _ = logit.size()
        logit = self.pred_map(logit)
        if gumbel is None:
            p = self.softmax(logit)
        else:
            #print 'logit ..............'
            #print tc.max((logit < 1e+10) == False)
            #print 'gumbel ..............'
            #print tc.max((gumbel < 1e+10) == False)
            #print 'aaa ..............'
            #aaa = (gumbel.add(logit)) / tao
            #print tc.max((aaa < 1e+10) == False)
            p = self.softmax((gumbel.add(logit)) / tao)
        p = p.view(d1, d2, self.output_size)

        return p

    def smoothingXentLoss1(self, pred_ll, target):

        # pred_ll (FloatTensor): batch_size*max_seq_len, n_classes
        # target  (LongTensor):  batch_size*max_seq_len
        if self.label_smoothing == 0.:
            # if label smoothing value is set to zero, the loss is equivalent to NLLLoss
            return self.criterion(ll, gold)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == PAD).unsqueeze(1), 0)
        #print pred_ll.size(), model_prob.size()
        xentropy = -(pred_ll * model_prob).sum()
        normalizing = -(self.confidence * math.log(self.confidence) + \
                        (self.output_size - 2) * self.smoothing_value * math.log(self.smoothing_value + 1e-20))

        return xentropy - normalizing

    def smoothingXentLoss(self, pred_ll, target):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -pred_ll.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.label_smoothing / pred_ll.size(-1)
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def embeddingLoss(self, prob_BLV, gold_BL, gold_mask_BL, bow_BN=None, bow_mask_BN=None):
        batch_size, max_L = gold_BL.size()
        E, bow_N = self.trg_word_emb.weight.size(1), bow_BN.size(1)
        gold_BLE = self.trg_word_emb(gold_BL) * gold_mask_BL[:, :, None]
        bow_BNE = self.trg_word_emb(bow_BN) * bow_mask_BN[:, :, None]
        bow_nE = bow_BNE[:, None, :, :].expand((-1, max_L, -1, -1)).contiguous().view(-1, E)
        gold_nE = gold_BLE[:, :, None, :].expand((-1, -1, bow_N, -1)).contiguous().view(-1, E)
        dist = F.pairwise_distance(bow_nE, gold_nE, p=2, keepdim=True)
        dist = dist.reshape(batch_size, max_L, bow_N).sum(-1)   # [B, L]
        pred_p_t = tc.gather(prob_BLV, dim=-1, index=gold_BL[:, :, None]).squeeze(-1) # [B, L] 
        pred_p_t = pred_p_t * gold_mask_BL  # [B, L]
        return ( pred_p_t * dist ).sum()

    def bowLoss_based_pred(self, pred_BLV, gold_mask_BL, bow_BN, bow_mask_BN):
        # p_b = sigmoid(sum_{t=1}^{M} s_t)
        bow_prob = self.sigmoid((pred_BLV * gold_mask_BL[:, :, None]).sum(1))   # [B, V]
        #bow_prob = self.softmax((pred_BLV * gold_mask_BL).sum(1), gold_mask_BL)
        bow_N = bow_BN.size(1)
        bow_ll_BNV = tc.log(bow_prob + 1e-20)[:, None, :].expand(-1, bow_N, -1)
        bow_ll_BNV = bow_ll_BNV * bow_mask_BN[:, :, None]
        bow_ll_flat_nV = bow_ll_BNV.view(-1, bow_ll_BNV.size(-1))
        bow_ce_loss = self.criterion(bow_ll_flat_nV, bow_BN.view(-1))
        return bow_ce_loss

    def forward(self, feed_BLO, gold_BL=None, gold_mask_BL=None, noise=None, bow_BN=None,
                bow_mask_BN=None, context_BLH=None):

        pred_BLV = self.pred_map(feed_BLO, noise)   # (batch_size, y_Lm1, out_size)
        # decoding, if gold is None and gold_mask is None:
        if gold_BL is None and gold_mask_BL is None:
            return -self.log_prob(pred_BLV)[-1] if wargs.self_norm_alpha is None else -pred_BLV

        assert pred_BLV.dim() == 3 and gold_BL.dim() == 2 and gold_mask_BL.dim() == 2, 'error'
        gold_flat_n, gold_mask_flat_n = gold_BL.view(-1), gold_mask_BL.view(-1)
        #print pred_2d.size(), pred_3d.size(), gold.size(), gold_mask.size(), bow.size(), bow_mask.size()
        ln_B1V, prob_BLV, ll_BLV = self.log_prob(pred_BLV, dim=-1)
        abs_logZ = (ln_B1V * gold_mask_BL[:, :, None]).abs().sum()
        ll_BLV = ll_BLV * gold_mask_BL[:, :, None]
        ll_flat_nV = ll_BLV.view(-1, ll_BLV.size(-1))

        # negative log likelihood, may be with label smoothing
        loss, ce_loss = self.smoothingXentLoss(ll_flat_nV, gold_flat_n)
        emb_loss, bow_loss = None, None

        if self.emb_loss is True:
            emb_loss = self.embeddingLoss(prob_BLV, gold_BL, gold_mask_BL, bow_BN, bow_mask_BN)
        if self.bow_loss is True:
            #bow_loss = self.bowLoss_based_pred(pred_BLV, gold_mask_BL, bow_BN, bow_mask_BN)
            pred_bow = self.map_vocab(context_BLH)  # (batch_size, y_Lm1, V)
            bow_loss = self.bowLoss_based_pred(pred_bow, gold_mask_BL, bow_BN, bow_mask_BN)

        pred_flat_nV = pred_BLV.view(-1, pred_BLV.size(-1))
        # ok prediction count in one minibatch
        ok_ytoks = (pred_flat_nV.max(dim=-1)[1]).eq(gold_flat_n).masked_select(gold_flat_n.ne(PAD)).sum()
        # final loss, xentropy
        return loss, ce_loss, emb_loss, bow_loss, ok_ytoks, abs_logZ

    '''
    Compute the loss in shards for efficiency
        outputs: the predict outputs from the model
        gold: correct target sentences in current batch
    '''
    def snip_back_prop(self, outputs, gold, gold_mask, bow, bow_mask, epo_idx, shard_size=100,
                       contexts=None):

        # (batch_size, y_Lm1, out_size)
        batch_nll, batch_ok_ytoks, batch_abs_logZ = 0, 0, 0
        word_norm = gold_mask.sum().item() if self.loss_norm == 'tokens' else gold.size(0)
        bow_norm = bow_mask.sum().item() if self.loss_norm == 'tokens' else bow.size(0)
        word_norm, bow_norm = float(word_norm), float(bow_norm)
        lambd = schedule_bow_lambda(epo_idx)
        if contexts is not None and self.bow_loss is False: contexts = contexts.detach()
        shard_state = { 'feed_BLO': outputs, 'gold_BL': gold, 'gold_mask_BL': gold_mask,
                       'bow_BN': bow, 'bow_mask_BN':bow_mask, 'context_BLH': contexts }

        for shard in shards(shard_state, shard_size):
            loss, ce_loss, emb_loss, bow_loss, ok_ytoks, abs_logZ = self(**shard)
            batch_nll += ce_loss.item()
            batch_ok_ytoks += ok_ytoks.item()
            batch_abs_logZ += abs_logZ.item()

            loss = loss.div(word_norm)
            if self.emb_loss is True:
                loss = loss + emb_loss.div(word_norm)
            elif self.bow_loss is True:
                loss = loss + lambd * bow_loss.div(bow_norm)
            loss.backward(retain_graph=True)

        return batch_nll, batch_ok_ytoks, batch_abs_logZ

def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, tc.Tensor):
                for v_chunk in tc.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    '''
    Args:
        state: A dictionary which corresponds to the output
               values for those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: if True, only yield the state, nothing else.
              otherwise, yield shards.
    Yields:
        each yielded shard is a dict.
    side effect:
        after the last shard, this function does back-propagation.
    '''
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values are not None
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, tc.Tensor) and state[k].requires_grad:
                variables.extend(zip(tc.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        tc.autograd.backward(inputs, grads)

