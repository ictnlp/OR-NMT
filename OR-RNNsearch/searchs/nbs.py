from __future__ import division

import sys
import copy
import time
import numpy as np
import torch as tc

import wargs
from tools.utils import *

class Nbs(object):

    def __init__(self, model, tvcb_i2w, k=10, ptv=None, noise=False,
                 print_att=False, batch_sample=False):

        self.k = k
        self.ptv = ptv
        self.noise = noise
        self.xs_mask = None
        self.tvcb_i2w = tvcb_i2w
        self.print_att = print_att
        self.C = [0] * 4
        self.batch_sample = batch_sample
        self.encoder, self.decoder, self.classifier = model.encoder, model.decoder, model.decoder.classifier
        debug('Batch sampling by beam search ... {}'.format(batch_sample))

    def beam_search_trans(self, x_BL, x_mask=None, y_mask=None):

        #print '-------------------- one sentence ............'
        self.trgs_len = y_mask.sum(0).data.int().tolist() if y_mask is not None else None
        if isinstance(x_BL, list): x_BL = tc.tensor(x_BL).long().unsqueeze(0)
        elif isinstance(x_BL, tuple): x_BL = x_BL[1].unsqueeze(0)
        self.B, self.srcL = x_BL.size(0), x_BL.size(1)
        if x_mask is None:
            x_mask = tc.ones((1, self.srcL), requires_grad=False)
            if wargs.gpu_id is not None: x_mask = x_mask.cuda()
        assert not ( self.batch_sample ^ (self.trgs_len is not None) ), 'sample ^ trgs_len'

        self.beam, self.hyps = [], [[] for _ in range(self.B)]
        self.batch_tran_cands = [[] for _ in range(self.B)]
        self.attent_probs = [[] for _ in range(self.B)] if self.print_att is True else None

        self.maxL = y_mask.size(1) if self.batch_sample is True else 2 * self.srcL
        # get initial state of decoder rnn and encoder context
        if wargs.gpu_id is not None and not x_BL.is_cuda: x_BL = x_BL.cuda()
        self.enc_src0 = self.encoder(x_BL, xs_mask=x_mask)
        self.s0, self.uh0 = self.decoder.init_state(self.enc_src0, xs_mask=x_mask)
        #if wargs.dec_layer_cnt > 1: self.s0 = [self.s0] * wargs.dec_layer_cnt
        # (1, trg_nhids), (1, src_len, src_nhids*2)
        init_beam(self.beam, cnt=self.maxL, s0=self.s0)

        if not wargs.with_batch:
            assert self.B == 1, 'Only support a sentences for batch'
            self.search()
        elif wargs.ori_search:   best_trans, best_loss = self.ori_batch_search()
        else:                    self.batch_search()
        # best_trans w/o <bos> and <eos> !!!

        #batch_tran_cands: [(trans, loss, attend)]
        for bidx in range(self.B):
            debug([ (a[0], a[1]) for a in self.batch_tran_cands[bidx] ])
            best_trans, best_loss = self.batch_tran_cands[bidx][0][0], self.batch_tran_cands[bidx][0][1]
            debug('Src[{}], maskL[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
                int(x_mask[:, bidx].sum()), self.srcL, len(best_trans), self.maxL, best_loss))
        debug('Average location of bp [{}/{}={:6.4f}]'.format(self.C[1], self.C[0], self.C[1] / self.C[0]))
        debug('Step[{}] stepout[{}]'.format(*self.C[2:]))

        #return filter_reidx(best_trans, self.tvcb_i2w), best_loss, attent_matrix
        return self.batch_tran_cands

    ##################################################################

    # Wen Zhang: beam search, no batch

    ##################################################################
    def search(self):

        self.true_bidx = [0]
        for i in range(1, self.maxL + 1):

            B_prevbs = self.beam[i - 1]
            prevb = B_prevbs[0]
            preb_sz = len(prevb)
            cnt_bp = (i >= 2)
            if cnt_bp: self.C[0] += preb_sz
            cands, preb_alpha = [], []
            for j in range(preb_sz):  # size of last beam
                # (45.32, (beam, trg_nhids), -1, 0)
                accum_im1, s_im1, _, y_im1, _ = prevb[j]

                a_i, s_i, y_im1, alpha_ij, _, _, _ = self.decoder.step(
                    s_im1, self.enc_src0, self.uh0, y_im1)
                self.C[2] += 1
                logit = self.decoder.step_out(y_im1, a_i, s_i)
                self.C[3] += 1
                # alpha_ij: (srcL, 1)
                preb_alpha.append(alpha_ij.squeeze(-1))

                next_ces = self.classifier(logit)
                next_ces = next_ces.cpu().data.numpy()
                next_ces_flat = next_ces.flatten()    # (1,vocsize) -> (vocsize,)
                ranks_idx_flat = part_sort(next_ces_flat, self.k - len(self.hyps[0]))
                k_avg_loss_flat = next_ces_flat[ranks_idx_flat]  # -log_p_y_given_x

                accum_i = accum_im1 + k_avg_loss_flat
                cands += [(accum_i[idx], s_i, 0, wid, j) for idx, wid in enumerate(ranks_idx_flat)]

            k_ranks_flat = part_sort(np.asarray(
                [cand[0] for cand in cands] + [np.inf]), self.k - len(self.hyps[0]))
            k_sorted_cands = [cands[r] for r in k_ranks_flat]
            if self.attent_probs is not None: self.attent_probs[0].append(tc.stack(preb_alpha, 1))

            next_beam_cur_sent = []
            for b in k_sorted_cands:
                if cnt_bp: self.C[1] += (b[-1] + 1)
                if b[-2] == EOS:
                    if wargs.len_norm: self.hyps[0].append(((b[0] / i), b[0]) + b[-3:] + (i,))
                    else: self.hyps[0].append((b[0], ) + b[-3:] + (i, ))
                    debug('Gen hypo {}'.format(self.hyps[0][-1]))
                    if len(self.hyps[0]) == self.k:
                        # output sentence, early stop, best one in k
                        debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                        sorted_hyps = sorted(self.hyps[0], key=lambda tup: tup[0])
                        for hyp in sorted_hyps: debug('{}'.format(hyp))
                        best_hyp = sorted_hyps[0]
                        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))
                        self.batch_tran_cands[0] = [back_tracking(self.beam, 0, hyp, \
                            self.attent_probs[0] if self.attent_probs is not None \
                                                    else None) for hyp in sorted_hyps]
                        return
                        #return back_tracking(self.beam, 0, best_hyp)
                # should calculate when generate item in current beam
                else: next_beam_cur_sent.append(b)
            self.beam[i] = [ next_beam_cur_sent ]

            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i][0]: debug(b[0:1] + b[-3:])    # do not output state
        n_remainings = len(self.beam[self.maxL])   # loop ends, how many sentences left

        return self.no_early_best(n_remainings)

    #@exeTime
    def batch_search(self):

        # s0: (B, trg_nhids), enc_src0: (B, srcL, src_nhids*2), uh0: (B, srcL, align_size)
        hyp_scores = np.zeros(1).astype('float32')
        delete_idx, prevb_id = None, None
        for i in range(1, self.maxL + 1):

            if all([len(a) > 0 for a in self.batch_tran_cands]) is True:
                debug('Early stop~ Normal beam search or sampling in this batch finished.')
                return
            B_prevbs = self.beam[i - 1]
            debug('\n{} Look Beam-{} {}'.format('-'*20, i - 1, '-'*20))
            for bidx, sent_prevb in enumerate(B_prevbs):
                debug('Sent {} '.format(bidx))
                for b in sent_prevb:    # do not output state
                    debug(b[0:1] + (None if b[1] is None else b[1].size(), b[-4].size()) + b[-3:])

            debug('\n{} Step-{} {}'.format('#'*20, i, '#'*20))
            '''
                [
                    batch_0 : [ (beam_item_0), (beam_item_1), ...,  ]
                    batch_1 : [ (beam_item_0), (beam_item_1), ...,  ]
                    batch_2 : [ (beam_item_0), (beam_item_1), ...,  ]
                    ......
                    batch_39: [ (beam_item_0), (beam_item_1), ...,  ]
                ]
            '''

            n_remainings = len(B_prevbs)
            enc_src, uh, y_im1, prebs_sz, hyp_scores, s_im1, self.true_bidx = \
                    [], [], [], [], [], [], []
            for bidx in range(n_remainings):
                prevb = B_prevbs[bidx]
                preb_sz = len(prevb)
                prebs_sz.append(preb_sz)
                hyp_scores += list(zip(*prevb))[0]
                s_im1 += list(zip(*prevb))[-4]
                y_im1 += list(zip(*prevb))[-2]
                self.true_bidx.append(prevb[0][-3])
                if self.enc_src0.dim() == 4:
                    # (L, L, 1, src_nhids) -> (L, L, preb_sz, src_nhids)
                    enc_src, uh = self.enc_src0.expand(1, 1, preb_sz, 1), self.uh0.expand(1, 1, preb_sz, 1)
                elif self.enc_src0.dim() == 3:
                    # (src_sent_len, B, src_nhids) -> (src_sent_len, B*preb_sz, src_nhids)
                    enc_src.append(self.enc_src0[bidx].repeat(preb_sz, 1, 1))
                    uh.append(self.uh0[bidx].repeat(preb_sz, 1, 1))
            enc_src, uh = tc.cat(enc_src, dim=1), tc.cat(uh, dim=1)

            cnt_bp = (i >= 2)
            if cnt_bp is True: self.C[0] += sum(prebs_sz)
            hyp_scores = np.array(hyp_scores)
            s_im1 = tc.stack(s_im1)

            def track_ys(cur_bidx):
                y_part_seqs = []
                for b in self.beam[cur_bidx - 1][0]:
                    seq, bp = [b[-2]], b[-1]
                    for i in reversed(range(0, cur_bidx - 1)):
                        _, _, _, _, w, backptr = self.beam[i][0][bp]
                        seq.append(w)
                        bp = backptr
                    y_part_seqs.append(seq[::-1])
                return y_part_seqs

            y_part_seqs = track_ys(i) # (preb_sz, trg_part_L)
            y_part_seqs = tc.tensor(y_part_seqs, requires_grad=False).view(-1, i)
            if wargs.gpu_id is not None: y_part_seqs = y_part_seqs.cuda()

            debug(y_im1)
            debug(y_part_seqs)
            y_im1 = self.decoder.trg_word_emb(y_part_seqs)[1][:, -1, :]
            step_output = self.decoder.step(s_im1, enc_src, uh, y_im1)
            a_i, s_i, y_im1, alpha_ij = step_output[:4]
            # a_i: (n_remainings*p, enc_hid_size), s_i: (n_remainings*p, dec_hid_size),
            # y_im1: (n_remainings*p, trg_wemb_size), alpha_ij: (n_remainings*p, srcL)

            self.C[2] += 1
            # (preb_sz, out_size)
            logit = self.decoder.step_out(y_im1, a_i, s_i)
            self.C[3] += 1

            # (n_remainings*prevb_sz, vocab_size)
            #wlog('bleu sampling, noise {}'.format(self.noise))
            next_ces = self.classifier(logit, noise=self.noise)
            next_ces = next_ces.cpu().data.numpy()
            voc_size = next_ces.shape[1]
            cand_scores = hyp_scores[:, None] + next_ces

            a, split_idx = 0, []
            for preb_sz in prebs_sz[:-1]:
                a += preb_sz
                split_idx.append(a)
            next_ces_B_prevb = np.split(cand_scores, split_idx, axis=0) # [B: (prevb, vocab)]
            debug(len(next_ces_B_prevb))
            _s_i, _alpha_ij = [], []
            for _idx in prebs_sz[:-1]:
                _s_i.append(s_i[:_idx])
                _alpha_ij.append(alpha_ij[:_idx].t())
                s_i, alpha_ij = s_i[_idx:], alpha_ij[_idx:]
            _s_i.append(s_i)    # [B: (prevb, dec_hid_size)]
            _alpha_ij.append(alpha_ij.t())  # # [B: (srcL, preb_sz)]
            next_step_beam, del_batch_idx = [], []
            import copy
            if self.batch_sample is True: _next_ces_B_prevb = copy.deepcopy(next_ces_B_prevb)  # bak
            for bidx in range(n_remainings):
                true_id, next_ces_prevb = self.true_bidx[bidx], next_ces_B_prevb[bidx]
                if self.attent_probs is not None: self.attent_probs[true_id].append(_alpha_ij[bidx])
                if len(self.hyps[true_id]) == self.k: continue
                #if len(self.hyps[true_id]) == self.k: continue  # have finished this sentence
                debug('Sent {}, {} hypos left ----'.format(bidx, self.k - len(self.hyps[true_id])))
                if self.batch_sample is True:
                    debug(next_ces_prevb.shape)
                    debug(next_ces_prevb[:, :8])
                    if i < self.trgs_len[true_id] - 1:
                        '''here we make the score of <e> so large that <e> can not be selected'''
                        next_ces_prevb[:, EOS] = [float('+inf')] * prebs_sz[bidx]
                    elif i == self.trgs_len[true_id] - 1:
                        '''here we make the score of <e> so large that <e> can not be selected'''
                        next_ces_prevb[:, EOS] = [float('-inf')] * prebs_sz[bidx]
                    else:
                        debug('Impossible ...')
                        import sys
                        sys.exit(0)
                    debug(next_ces_prevb[:, :8])
                cand_scores_flat = next_ces_prevb.flatten()
                ranks_flat = part_sort(cand_scores_flat, self.k - len(self.hyps[true_id]))
                prevb_id = ranks_flat // voc_size
                #debug('For beam [{}], pre-beam ids: {}'.format(i, prevb_id))
                word_indices = ranks_flat % voc_size
                costs = cand_scores_flat[ranks_flat] if self.batch_sample is False else \
                        _next_ces_B_prevb[bidx].flatten()[ranks_flat]
                next_beam_cur_sent = []
                for _j, b in enumerate(zip(costs, _s_i[bidx][prevb_id], [true_id]*len(prevb_id), word_indices, prevb_id)):
                    delete_idx = []
                    bp = b[-1]
                    if wargs.len_norm == 0: score = (b[0], None)
                    elif wargs.len_norm == 1: score = (b[0] / i, b[0])
                    elif wargs.len_norm == 2:   # alpha length normal
                        lp, cp = lp_cp(bp, i, bidx, self.beam)
                        score = (b[0] / lp + cp, b[0])
                    if cnt_bp: self.C[1] += (bp + 1)
                    if b[-2] == EOS:
                        #assert self.batch_sample is False, 'Impossible ...'
                        delete_idx.append(bp)
                        self.hyps[true_id].append(score + b[-3:] + (i, ))   # contains <b> and <e>
                        debug('Gen hypo {} {} {}'.format(bidx, true_id, self.hyps[true_id][-1]))
                        # because i starts from 1, so the length of the first beam is 1, no <bos>
                        if len(self.hyps[true_id]) == self.k:
                            # output sentence, early stop, best one in k
                            debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                            sorted_hyps = sorted(self.hyps[true_id], key=lambda tup: tup[0])
                            for hyp in sorted_hyps: debug('{}'.format(hyp))
                            #best_hyp = sorted_hyps[0]
                            #debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))
                            del_batch_idx.append(bidx)
                            self.batch_tran_cands[true_id] = [back_tracking(self.beam, bidx, hyp,\
                                self.attent_probs[true_id] if self.attent_probs is not None \
                                                        else None) for hyp in sorted_hyps]
                    # should calculate when generate item in current beam
                    else:
                        if wargs.len_norm == 2:
                            next_beam_cur_sent.append((b[0], _alpha_ij[bidx][:, bp]) + b[1:])
                        else: next_beam_cur_sent.append(b)
                if len(next_beam_cur_sent) > 0: next_step_beam.append(next_beam_cur_sent)
            self.beam[i] = next_step_beam

            if len(del_batch_idx) < n_remainings:
                self.enc_src0 = self.enc_src0[list(filter(lambda x: x not in del_batch_idx, range(n_remainings)))]
                self.uh0 = self.uh0[list(filter(lambda x: x not in del_batch_idx, range(n_remainings)))]
            del enc_src, uh, y_im1, y_part_seqs, s_im1   # free the tensor

        # no early stop, back tracking
        n_remainings = len(self.beam[self.maxL])   # loop ends, how many sentences left
        self.no_early_best(n_remainings)

    def no_early_best(self, n_remainings):

        if n_remainings == 0: return
        debug('==Start== No early stop ...')
        for bidx in range(n_remainings):
            true_id = self.true_bidx[bidx]
            debug('Sent {}, true id {}'.format(bidx, true_id))
            hyps = self.hyps[true_id]
            if len(hyps) == self.k: continue  # have finished this sentence
            # no early stop, back tracking
            if len(hyps) == 0:
                debug('No early stop, no hyp with EOS, select k hyps length {} '.format(self.maxL))
                for hyp in self.beam[self.maxL][bidx]:
                    if wargs.len_norm == 0: score = (hyp[0], None)
                    elif wargs.len_norm == 1: score = (hyp[0] / self.maxL, hyp[0])
                    elif wargs.len_norm == 2:   # alpha length normal
                        lp, cp = lp_cp(hyp[-1], self.maxL, bidx, self.beam)
                        score = (hyp[0] / lp + cp, hyp[0])
                    hyps.append(score + hyp[-3:] + (self.maxL, ))
            else:
                debug('No early stop, no enough {} hyps with EOS, select the best '
                      'one from {} hyps.'.format(self.k, len(hyps)))
                #best_hyp = sorted_hyps[0]
            sorted_hyps = sorted(hyps, key=lambda tup: tup[0])
            for hyp in sorted_hyps: debug('{}'.format(hyp))
            debug('Sent {}: Best hyp length (w/ EOS)[{}]'.format(bidx, sorted_hyps[0][-1]))
            self.batch_tran_cands[true_id] = [back_tracking(self.beam, bidx, hyp, \
                        self.attent_probs[true_id] if self.attent_probs is not None \
                                                else None) for hyp in sorted_hyps]
        debug('==End== No early stop ...')

    def ori_batch_search(self):

        sample = []
        sample_score = []

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []

        # s0: (B, trg_nhids), enc_src0: (B, srcL, src_nhids*2), uh0: (B, srcL, align_size)
        s_im1, y_im1 = self.s0, [BOS]  # indicator for the first target word (bos target)
        preb_sz = 1

        for ii in range(self.maxL):

            cnt_bp = (ii >= 1)
            if cnt_bp: self.C[0] += preb_sz
            # (src_sent_len, 1, 2*src_nhids) -> (src_sent_len, live_k, 2*src_nhids)
            enc_src, uh = self.enc_src0.repeat(live_k, 1, 1), self.uh0.repeat(live_k, 1, 1)

            #c_i, s_i = self.decoder.step(c_im1, enc_src, uh, y_im1)
            a_i, s_im1, y_im1, _ = self.decoder.step(s_im1, enc_src, uh, y_im1)
            self.C[2] += 1
            # (preb_sz, out_size)
            # logit = self.decoder.logit(s_i)
            logit = self.decoder.step_out(y_im1, a_i, s_im1)
            self.C[3] += 1
            next_ces = self.classifier(logit)
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
            for idx in range(len(new_hyp_samples)):
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
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

        if wargs.len_norm:
            lengths = numpy.array([len(s) for s in sample])
            sample_score = sample_score / lengths
        sidx = numpy.argmin(sample_score)

        return sample[sidx], sample_score[sidx]



