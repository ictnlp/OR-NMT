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
        s0, enc_src0, uh0 = self.model.init(s_tensor)
        #if wargs.dec_layer_cnt > 1: s0 = [s0] * wargs.dec_layer_cnt

        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        init_beam(self.beam, cnt=self.maxL, hs0=enc_src0, s0=s0, detail=True)

        best_trans, best_loss = self.batch_search()
        # best_trans w/o <bos> and <eos> !!!

        debug('Src[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
            self.srcL, len(best_trans), self.maxL, best_loss))

        debug('Average location of bp [{}/{}={:6.4f}]'.format(
            self.C[1], self.C[0], self.C[1] / self.C[0]))
        debug('Step[{}] stepout[{}]'.format(*self.C[2:]))

        return filter_reidx(best_trans, self.tvcb_i2w), best_loss

    #@exeTime
    def batch_search(self):

        # s0: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)
        hyp_scores = np.zeros(1).astype('float32')

        for i in range(1, self.maxL + 1):

            prevb = self.beam[i - 1]
            preb_sz = len(prevb)
            cnt_bp = (i >= 2)
            if cnt_bp: self.C[0] += preb_sz

            # (slen, 1, enc_hid_size) -> (slen, enc_hid_size) -> (slen, preb_sz, enc_hid_size)
            hs_im1 = tc.stack([b[1].squeeze(1) for b in prevb], dim=1)
            uh_im1 = self.model.ha(hs_im1)

            # batch states of previous beam, (1, nhids) -> (preb_sz, 1, nhids) -> (preb_sz, nhids)
            s_im1 = tc.stack([b[2] for b in prevb], dim=0).squeeze(1)
            #c_im1 = [tc.stack(tuple([prevb[bid][1][lid] for bid in range(len(prevb))])
            #                 ).squeeze(1) for lid in range(len(prevb[0][1]))]
            y_im1 = [b[-2] for b in prevb]
            # (src_sent_len, 1, src_nhids) -> (src_sent_len, preb_sz, src_nhids)

            #c_i, s_i = self.decoder.step(c_im1, enc_src, uh, y_im1)
            a_i, s_i, y_im1, alpha_ij = self.decoder.step(s_im1, hs_im1, uh_im1, y_im1)
            self.C[2] += 1
            hs_im1 = self.decoder.write_attention(alpha_ij, s_i, hs_im1)
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
            hs_im1 = [hs_im1[:, idx, :].unsqueeze(1) for idx in prevb_id]
            #for b in zip(costs, batch_ci, word_indices, prevb_id):
            for b in zip(costs, hs_im1, s_i[tp_bid], word_indices, prevb_id):
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

