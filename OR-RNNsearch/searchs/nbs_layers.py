from __future__ import division

import sys
import numpy as np
import torch as tc
from torch.autograd import Variable

import wargs
from tools.utils import *

class Func(object):

    def __init__(self, lqc):

        self.lqc = lqc

class NBS(Func):

    def __init__(self, model, tvcb_i2w=None, k=10, ptv=None):

        self.lqc = [0] * 10
        super(NBS, self).__init__(self.lqc)

        self.model = model
        self.decoder = model.decoder

        self.tvcb_i2w = tvcb_i2w
        self.k = k
        self.ptv = ptv

    def beam_search_trans(self, s_list):

        self.locrt = [0] * 2
        self.beam = []
        self.translations = []

        self.maxlen = 2 * len(s_list)

        s_tensor = tc.Tensor(s_list).long().unsqueeze(-1)
        #s_tensor = self.model.src_lookup_table(Variable(s_tensor, volatile=True))

        best_trans, loss = self.beam_search_comb(s_tensor) if \
                wargs.with_batch else self.beam_search(s_tensor)

        debug('@source[{}], translation(without eos)[{}], maxlen[{}], loss[{}]'.format(
            len(s_list), len(best_trans), self.maxlen, loss))

        return filter_reidx(best_trans, self.tvcb_i2w)

    ##################################################################

    # Wen Zhang: beam search, no batch

    ##################################################################
    def beam_search(self, s_tensor):

        s_init, enc_src, uh = self.model.init(s_tensor)

        maxlen = self.maxlen
        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        init_beam(self.beam, cnt=maxlen, s0=s_init)

        for i in range(1, self.maxlen + 1):
            if (i - 1) % 10 == 0:
                debug(str(i - 1))
            cands = []
            for j in xrange(len(self.beam[i - 1])):  # size of last beam
                # (45.32, (beam, trg_nhids), -1, 0)
                accum_im1, s_im1, y_im1, bp_im1 = self.beam[i - 1][j]

                assert isinstance(y_im1, int)
                y_im1 = tc.Tensor([y_im1]).long().unsqueeze(-1)
                if wargs.gpu_id: y_im1 = y_im1.cuda()
                y_im1 = Variable(y_im1, requires_grad=False, volatile=True)
                y_im1 = self.decoder.trg_lookup_table(y_im1)
                s_i, c_i = self.decoder.step(s_im1, enc_src, uh, y_im1)
                logit = self.decoder.logit(s_i, y_im1, c_i)

                next_ces = self.model.classifier(logit)
                next_ces = next_ces.cpu().data.numpy()
                next_ces_flat = next_ces.flatten()    # (1,vocsize) -> (vocsize,)
                ranks_idx_flat = part_sort(next_ces_flat, self.k - len(self.translations))
                k_avg_loss_flat = next_ces_flat[ranks_idx_flat]  # -log_p_y_given_x

                accum_i = accum_im1 + k_avg_loss_flat
                cands += [(accum_i[idx], s_i, wid, j)
                          for idx, wid in enumerate(ranks_idx_flat)]

            k_ranks_flat = part_sort(np.asarray(
                [cand[0] for cand in cands] + [np.inf]), self.k - len(self.translations))
            k_sorted_cands = [cands[r] for r in k_ranks_flat]

            for b in k_sorted_cands:
                if b[-2] == EOS:
                    debug('add: {}'.format(((b[0] / i), b[0]) + b[-2:] + (i,)))
                    if wargs.with_norm:
                        self.translations.append(((b[0] / i), b[0]) + b[-2:] + (i,))
                    else:
                        self.translations.append((b[0], ) + b[-2:] + (i, ))
                    if len(self.translations) == self.k:
                        # output sentence, early stop, best one in k
                        debug('early stop! see {} samples ending with EOS.'.format(self.k))
                        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
                        debug('average location of back pointers [{}/{}={}]'.format(
                            self.locrt[0], self.locrt[1], avg_bp))
                        sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                        best_sample = sorted_samples[0]
                        debug('translation length(with EOS) [{}]'.format(best_sample[-1]))
                        for sample in sorted_samples:  # tuples
                            debug('{}'.format(sample))

                        return back_tracking(self.beam, best_sample)
                else:
                    # should calculate when generate item in current beam
                    self.locrt[0] += (b[-1] + 1)
                    self.locrt[1] += 1
                    self.beam[i].append(b)
            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i]:
                debug(b[0:1] + b[2:])    # do not output state

        # no early stop, back tracking
        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
        debug('average location of back pointers [{}/{}={}]'.format(
            self.locrt[0], self.locrt[1], avg_bp))
        if len(self.translations) == 0:
            debug('no early stop, no candidates ends with EOS, selecting from '
                'len {} candidates, may not end with EOS.'.format(maxlen))
            if wargs.with_norm:
                best_sample = (self.beam[maxlen][0][0], self.beam[maxlen][0][0]) + \
                        self.beam[maxlen][0][-2:] + (maxlen, )
            else:
                best_sample = (self.beam[maxlen][0][0],) + self.beam[maxlen][0][-2:] + (maxlen, )
            debug('translation length(with EOS) [{}]'.format(best_sample[-1]))
            return back_tracking(self.beam, best_sample)
        else:
            debug('no early stop, not enough {} candidates end with EOS, selecting the best '
                'sample ending with EOS from {} samples.'.format(self.k, len(self.translations)))
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            debug('translation length(with EOS) [{}]'.format(best_sample[-1]))
            for sample in sorted_samples:  # tuples
                debug('{}'.format(sample))
            return back_tracking(self.beam, best_sample)

    #@exeTime
    def beam_search_comb(self, s_tensor):

        s_init, enc_src0, uh0 = self.model.init(s_tensor)
        # s_init: 1x512
        slen, enc_size, align_size = enc_src0.size(0), enc_src0.size(2), uh0.size(2)

        maxlen = self.maxlen
        hyp_scores = np.zeros(1).astype('float32')

        if wargs.dec_layer_cnt > 1: s_init = [s_init] * wargs.dec_layer_cnt
        init_beam(self.beam, cnt=maxlen, s0=s_init)

        for i in range(1, self.maxlen + 1):
            if (i - 1) % 10 == 0: debug(str(i - 1))

            prevb = self.beam[i - 1]
            preb_sz = len(prevb)
            # batch states of previous beam, (preb_sz, 1, nhids) -> (preb_sz, nhids)
            #s_im1 = tc.stack(tuple([b[1] for b in prevb]), dim=0).squeeze(1)
            c_im1 = [tc.stack(tuple([prevb[bid][1][lid] for bid in range(len(prevb))])
                             ).squeeze(1) for lid in range(len(prevb[0][1]))]
            y_im1 = [b[2] for b in prevb]
            # (src_sent_len, 1, src_nhids) -> (src_sent_len, preb_sz, src_nhids)

            enc_src = enc_src0.view(slen, -1, enc_size).expand(slen, preb_sz, enc_size)
            uh = uh0.view(slen, -1, align_size).expand(slen, preb_sz, align_size)

            c_i, s_i = self.decoder.step(c_im1, enc_src, uh, y_im1)
            # (preb_sz, out_size)
            logit = self.decoder.logit(s_i)

            # (preb_sz, vocab_size)
            next_ces = self.model.classifier(logit)
            next_ces = next_ces.cpu().data.numpy()
            #next_ces = -next_scores if self.ifscore else self.fn_ce(next_scores)
            cand_scores = hyp_scores[:, None] + next_ces
            cand_scores_flat = cand_scores.flatten()
            ranks_flat = part_sort(cand_scores_flat, self.k - len(self.translations))
            voc_size = next_ces.shape[1]
            prevb_id = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_scores_flat[ranks_flat]

            batch_ci = [[c_i[lid][bid].unsqueeze(0) for
                         lid in range(len(c_i))] for bid in prevb_id]
            #tp_bid = tc.from_numpy(prevb_id).cuda() if wargs.gpu_id else tc.from_numpy(prevb_id)
            for b in zip(costs, batch_ci, word_indices, prevb_id):
                if b[-2] == EOS:
                    if wargs.with_norm:
                        self.translations.append(((b[0] / i), b[0]) + b[2:] + (i, ))
                    else:
                        self.translations.append((b[0], ) + b[2:] + (i,))
                    if len(self.translations) == self.k:
                        # output sentence, early stop, best one in k
                        debug('early stop! see {} samples ending with EOS.'.format(self.k))
                        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
                        debug('average location of back pointers [{}/{}={}]'.format(
                            self.locrt[0], self.locrt[1], avg_bp))
                        sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                        best_sample = sorted_samples[0]
                        debug('translation length(with EOS) [{}]'.format(best_sample[-1]))
                        for sample in sorted_samples:  # tuples
                            debug('{}'.format(sample))
                        return back_tracking(self.beam, best_sample)
                else:
                    # should calculate when generate item in current beam
                    self.locrt[0] += (b[-1] + 1)
                    self.locrt[1] += 1
                    self.beam[i].append(b)
            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i]:
                debug(b[0:1] + b[2:])    # do not output state
            hyp_scores = np.array([b[0] for b in self.beam[i]])

        # no early stop, back tracking
        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
        debug('average location of back pointers [{}/{}={}]'.format(
            self.locrt[0], self.locrt[1], avg_bp))
        if len(self.translations) == 0:
            debug('no early stop, no candidates ends with EOS, selecting from '
                'len {} candidates, may not end with EOS.'.format(maxlen))
            if wargs.with_norm:
                best_sample = (self.beam[maxlen][0][0] / maxlen, self.beam[maxlen][0][0]) + \
                        self.beam[maxlen][0][-2:] + (maxlen, )
            else:
                best_sample = (self.beam[maxlen][0][0],) + self.beam[maxlen][0][-2:] + (maxlen, )
            debug('translation length(with EOS) [{}]'.format(best_sample[-1]))
            return back_tracking(self.beam, best_sample)
        else:
            debug('no early stop, not enough {} candidates end with EOS, selecting the best '
                'sample ending with EOS from {} samples.'.format(self.k, len(self.translations)))
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            debug('translation length(with EOS) [{}]'.format(best_sample[-1]))
            for sample in sorted_samples:  # tuples
                debug('{}'.format(sample))
            return back_tracking(self.beam, best_sample)


