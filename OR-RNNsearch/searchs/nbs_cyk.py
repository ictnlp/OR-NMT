from __future__ import division

import sys
import numpy as np
import torch as tc
from torch.autograd import Variable

import wargs
from tools.utils import *

class Nbs(object):

    def __init__(self, model, tvcb_i2w, k=10, ptv=None, noise=False):

        self.model = model
        self.decoder = model.decoder

        self.tvcb_i2w = tvcb_i2w
        self.k = k
        self.ptv = ptv
        self.noise = noise

        self.C = [0] * 4

    def beam_search_trans(self, s_list):

        self.srcL = len(s_list)
        self.maxL = 2 * self.srcL

        self.beam, self.hyps = [], []

        s_tensor = tc.Tensor(s_list).long().unsqueeze(-1)
        #s_tensor = self.model.src_lookup_table(Variable(s_tensor, volatile=True))

        # get initial state of decoder rnn and encoder context
        # s_tensor: (srcL, batch_size), batch_size==beamsize==1
        #self.s0, self.enc_src0, self.uh0 = self.model.init(s_tensor, test=True)
        self.s0, self.enc_src0, self.uh0, self.xs0_cyk, self.uh0_cyk, self.cykmask0 = self.model.init(s_tensor, test=True)
        #if wargs.dec_layer_cnt > 1: self.s0 = [self.s0] * wargs.dec_layer_cnt

        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        init_beam(self.beam, cnt=self.maxL, s0=self.s0)

        if not wargs.with_batch: best_trans, best_loss = self.search()
        elif wargs.ori_search:   best_trans, best_loss = self.ori_batch_search()
        else:                    best_trans, best_loss = self.batch_search()
        # best_trans w/o <bos> and <eos> !!!

        debug('Src[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
            self.srcL, len(best_trans), self.maxL, best_loss))

        debug('Average location of bp [{}/{}={:6.4f}]'.format(
            self.C[1], self.C[0], self.C[1] / self.C[0]))
        debug('Step[{}] stepout[{}]'.format(*self.C[2:]))

        return filter_reidx(best_trans, self.tvcb_i2w), best_loss

    ##################################################################

    # Wen Zhang: beam search, no batch

    ##################################################################
    def search(self):

        for i in range(1, self.maxL + 1):

            prevb = self.beam[i - 1]
            preb_sz = len(prevb)
            cnt_bp = (i >= 2)
            if cnt_bp: self.C[0] += preb_sz
            cands = []
            for j in xrange(preb_sz):  # size of last beam
                # (45.32, (beam, trg_nhids), -1, 0)
                accum_im1, s_im1, y_im1, bp_im1 = self.beam[i - 1][j]

                a_i, s_i, y_im1, _ = self.decoder.step(s_im1, self.enc_src0, self.uh0, y_im1)
                self.C[2] += 1
                logit = self.decoder.step_out(s_i, y_im1, a_i)
                self.C[3] += 1

                next_ces = self.model.classifier(logit)
                next_ces = next_ces.cpu().data.numpy()
                next_ces_flat = next_ces.flatten()    # (1,vocsize) -> (vocsize,)
                ranks_idx_flat = part_sort(next_ces_flat, self.k - len(self.hyps))
                k_avg_loss_flat = next_ces_flat[ranks_idx_flat]  # -log_p_y_given_x

                accum_i = accum_im1 + k_avg_loss_flat
                cands += [(accum_i[idx], s_i, wid, j) for idx, wid in enumerate(ranks_idx_flat)]

            k_ranks_flat = part_sort(np.asarray(
                [cand[0] for cand in cands] + [np.inf]), self.k - len(self.hyps))
            k_sorted_cands = [cands[r] for r in k_ranks_flat]

            for b in k_sorted_cands:
                if cnt_bp: self.C[1] += (b[-1] + 1)
                if b[-2] == EOS:
                    if wargs.len_norm: self.hyps.append(((b[0] / i), b[0]) + b[-2:] + (i,))
                    else: self.hyps.append((b[0], ) + b[-2:] + (i, ))
                    debug('Gen hypo {}'.format(self.hyps[-1]))
                    if len(self.hyps) == self.k:
                        # output sentence, early stop, best one in k
                        debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                        sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
                        for hyp in sorted_hyps: debug('{}'.format(hyp))
                        best_hyp = sorted_hyps[0]
                        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))
                        return back_tracking(self.beam, best_hyp)
                # should calculate when generate item in current beam
                else: self.beam[i].append(b)

            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i]: debug(b[0:1] + b[2:])    # do not output state

        return back_tracking(self.beam, self.no_early_best())

    #@exeTime
    def batch_search(self):

        # s0: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)
        slen, enc_size, align_size = self.srcL, self.enc_src0.size(-1), self.uh0.size(-1)
        hyp_scores = np.zeros(1).astype('float32')

        for i in range(1, self.maxL + 1):

            prevb = self.beam[i - 1]
            preb_sz = len(prevb)
            cnt_bp = (i >= 2)
            if cnt_bp: self.C[0] += preb_sz
            # batch states of previous beam, (preb_sz, 1, nhids) -> (preb_sz, nhids)
            #for b in prevb: print b[1].size()
            s_im1 = tc.stack(tuple([b[1] for b in prevb]), dim=0).squeeze(1)
            #print s_im1.size()
            #c_im1 = [tc.stack(tuple([prevb[bid][1][lid] for bid in range(len(prevb))])
            #                 ).squeeze(1) for lid in range(len(prevb[0][1]))]
            y_im1 = [b[2] for b in prevb]
            # (src_sent_len, 1, src_nhids) -> (src_sent_len, preb_sz, src_nhids)
            enc_src = self.enc_src0.view(slen, -1, enc_size).expand(slen, preb_sz, enc_size)
            uh = self.uh0.view(slen, -1, align_size).expand(slen, preb_sz, align_size)

            # (L, L, 1, src_nhids) -> (L, L, preb_sz, src_nhids)
            xs_cyk = self.xs0_cyk.view(slen, slen, -1, enc_size).expand(slen, slen, preb_sz, enc_size)
            uh_cyk = self.uh0_cyk.view(slen, slen, -1, align_size).expand(slen, slen, preb_sz, align_size)

            #c_i, s_i = self.decoder.step(c_im1, enc_src, uh, y_im1)
            a_i, s_i, y_im1, _ = self.decoder.step(s_im1, enc_src, uh, y_im1, xs_cyk, uh_cyk,
                                                   cykmask=self.cykmask0)
            self.C[2] += 1
            # (preb_sz, out_size)
            # logit = self.decoder.logit(s_i)
            logit = self.decoder.step_out(s_i, y_im1, a_i)
            self.C[3] += 1

            # (preb_sz, vocab_size)
            next_ces = self.model.classifier(logit)
            next_ces = next_ces.cpu().data.numpy()
            #next_ces = -next_scores if self.ifscore else self.fn_ce(next_scores)
            cand_scores = hyp_scores[:, None] + next_ces
            cand_scores_flat = cand_scores.flatten()
            ranks_flat = part_sort(cand_scores_flat, self.k - len(self.hyps))
            voc_size = next_ces.shape[1]
            prevb_id = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_scores_flat[ranks_flat]

            #batch_ci = [[c_i[lid][bid].unsqueeze(0) for
            #             lid in range(len(c_i))] for bid in prevb_id]
            tp_bid = tc.from_numpy(prevb_id).cuda() if wargs.gpu_id else tc.from_numpy(prevb_id)
            #for b in zip(costs, batch_ci, word_indices, prevb_id):
            #print type(s_i[tp_bid])
            #print s_i[tp_bid].size()
            for b in zip(costs, s_i[tp_bid], word_indices, prevb_id):
                if cnt_bp: self.C[1] += (b[-1] + 1)
                if b[-2] == EOS:
                    if wargs.len_norm: self.hyps.append(((b[0] / i), b[0]) + b[-2:] + (i, ))
                    else: self.hyps.append((b[0], ) + b[-2:] + (i,))
                    debug('Gen hypo {}'.format(self.hyps[-1]))
                    # because i starts from 1, so the length of the first beam is 1, no <bos>
                    if len(self.hyps) == self.k:
                        # output sentence, early stop, best one in k
                        debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                        sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
                        for hyp in sorted_hyps: debug('{}'.format(hyp))
                        best_hyp = sorted_hyps[0]
                        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))

                        return back_tracking(self.beam, best_hyp)
                # should calculate when generate item in current beam
                else: self.beam[i].append(b)

            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i]: debug(b[0:1] + b[2:])    # do not output state
            hyp_scores = np.array([b[0] for b in self.beam[i]])

        # no early stop, back tracking
        return back_tracking(self.beam, self.no_early_best())

    def no_early_best(self):

        # no early stop, back tracking
        if len(self.hyps) == 0:
            debug('No early stop, no hyp ending with EOS, select one length {} '.format(self.maxL))
            best_hyp = self.beam[self.maxL][0]
            if wargs.len_norm:
                best_hyp = (best_hyp[0]/self.maxL, best_hyp[0], ) + best_hyp[-2:] + (self.maxL, )
            else:
                best_hyp = (best_hyp[0], ) + best_hyp[-2:] + (self.maxL, )

        else:
            debug('No early stop, no enough {} hyps ending with EOS, select the best '
                  'one from {} hyps.'.format(self.k, len(self.hyps)))
            sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
            for hyp in sorted_hyps: debug('{}'.format(hyp))
            best_hyp = sorted_hyps[0]

        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))

        return best_hyp


    def ori_batch_search(self):

        sample = []
        sample_score = []

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []

        # s0: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)
        slen, enc_size, align_size = self.srcL, self.enc_src0.size(2), self.uh0.size(2)

        s_im1, y_im1 = self.s0, [BOS]  # indicator for the first target word (bos target)
        preb_sz = 1

        for ii in xrange(self.maxL):

            cnt_bp = (ii >= 1)
            if cnt_bp: self.C[0] += preb_sz
            # (src_sent_len, 1, 2*src_nhids) -> (src_sent_len, live_k, 2*src_nhids)
            enc_src = self.enc_src0.view(slen, -1, enc_size).expand(slen, live_k, enc_size)
            uh = self.uh0.view(slen, -1, align_size).expand(slen, live_k, align_size)

            #c_i, s_i = self.decoder.step(c_im1, enc_src, uh, y_im1)
            a_i, s_im1, y_im1, _ = self.decoder.step(s_im1, enc_src, uh, y_im1)
            self.C[2] += 1
            # (preb_sz, out_size)
            # logit = self.decoder.logit(s_i)
            logit = self.decoder.step_out(s_im1, y_im1, a_i)
            self.C[3] += 1
            next_ces = self.model.classifier(logit)
            next_ces = next_ces.cpu().data.numpy()
            #cand_scores = hyp_scores[:, None] - numpy.log(next_scores)
            cand_scores = hyp_scores[:, None] + next_ces
            cand_flat = cand_scores.flatten()
            # ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            # we do not need to generate k candidate here, because we just need to generate k-dead_k
            # more candidates ending with eos, so for each previous candidate we just need to expand
            # k-dead_k candidates
            ranks_flat = part_sort(cand_flat, self.k - dead_k)
            # print ranks_flat, cand_flat[ranks_flat[1]], cand_flat[ranks_flat[8]]

            voc_size = next_ces.shape[1]
            trans_indices = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(self.k - dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                if cnt_bp: self.C[1] += (ti + 1)
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = costs[idx]
                new_hyp_states.append(s_im1[ti])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            # current beam, if the hyposise ends with eos, we do not
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == EOS:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    # print new_hyp_scores[idx], new_hyp_samples[idx]
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1: break
            if dead_k >= self.k: break

            preb_sz = len(hyp_states)
            y_im1 = [w[-1] for w in hyp_samples]
            s_im1 = tc.stack(hyp_states, dim=0)

        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

        if wargs.len_norm:
            lengths = numpy.array([len(s) for s in sample])
            sample_score = sample_score / lengths
        sidx = numpy.argmin(sample_score)

        return sample[sidx], sample_score[sidx]



