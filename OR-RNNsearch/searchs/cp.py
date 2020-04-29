from __future__ import division

import sys
import time
import copy
import heapq
import numpy
import torch.nn as nn
from itertools import count
from collections import OrderedDict

from tools.utils import *

class Wcp(object):

    def __init__(self, model, tvcb_i2w=None, k=10, thresh=100.0, lm=None,
                 ngram=3, ptv=None, print_att=False):

        self.model = model
        self.decoder = model.decoder
        self.classifier = self.decoder.classifier

        self.k = k
        self.lm = lm
        self.ptv = ptv
        self.ngram = ngram
        self.thresh = thresh
        self.tvcb_i2w = tvcb_i2w
        self.print_att = print_att

        self.C = [0] * 6
        self.pdist = nn.PairwiseDistance(2)

    def cube_prune_trans(self, x_LB, x_mask=None):

        self.cnt = count()
        if isinstance(x_LB, list): x_LB = tc.Tensor(x_LB).long().unsqueeze(-1)
        self.srcL, self.B = x_LB.size()
        self.maxL = 2 * self.srcL
        if x_mask is None:
            x_mask = tc.ones((self.srcL, 1))
            x_mask = Variable(x_mask, requires_grad=False, volatile=True).cuda()

        self.beam, self.hyps = [], [[] for _ in range(self.B)]
        self.batch_tran_cands = [[] for _ in range(self.B)]
        self.attent_probs = [[] for _ in range(self.B)] if self.print_att is True else None

        # s_tensor: (len, 1), beamsize==1
        s_init, self.enc_src0, self.uh0 = self.model.init(x_LB, xs_mask=x_mask, test=True)
        # s_init: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)

        init_beam(self.beam, cnt=self.maxL, s0=s_init, cp=True)

        '''
        best_trans, best_loss = self.cube_pruning()

        debug('Src[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
            srcL, len(best_trans), self.maxL, best_loss))

        return filter_reidx(best_trans, self.tvcb_i2w), best_loss
        '''

        self.cube_pruning()

        #batch_tran_cands: [(trans, loss, attend)]
        for bidx in range(self.B):
            debug([ (a[0], a[1]) for a in self.batch_tran_cands[bidx] ])
            best_trans, best_loss = self.batch_tran_cands[bidx][0][0], self.batch_tran_cands[bidx][0][1]
            debug('Src[{}], maskL[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
                x_mask[:, bidx].sum().data[0], self.srcL, len(best_trans), self.maxL, best_loss))
        debug('Average Merging Rate [{}/{}={:6.4f}]'.format(self.C[1], self.C[0], self.C[1] / self.C[0]))
        debug('Average location of bp [{}/{}={:6.4f}]'.format(self.C[3], self.C[2], self.C[3] / self.C[2]))
        debug('Step[{}] stepout[{}]'.format(*self.C[4:]))

        return self.batch_tran_cands


    ##################################################################

    # NOTE: merge all candidates in previous beam by euclidean distance of two state vector or KL
    # distance of alignment probability

    ##################################################################

    def merge(self, bidx, eq_classes):

        B_prevbs = self.beam[bidx - 1]
        prevb = B_prevbs[0]
        ye_im1 = tc.Tensor([k[-2] for k in prevb]).long()
        if wargs.gpu_id: ye_im1 = ye_im1.cuda()
        ye_im1 = Variable(ye_im1, requires_grad=False, volatile=True)
        ye_im1 = self.decoder.trg_lookup_table(ye_im1)
        len_prevb = len(prevb)
        used = []
        key = 0

        _memory = [None] * len_prevb
        for j in range(len_prevb):  # index of each item in last beam

            if j in used: continue

            tmp = []
            if _memory[j]:
                _needed = _memory[j]
                score_im1_1, p_ij_1, s_im1_1, y_im1_1, ye_im1_1, nj = _needed
                assert(j == nj)
            else:
                score_im1_1, p_ij_1, s_im1_1, _, y_im1_1, _ = prevb[j]
                ye_im1_1 = ye_im1[j]
                _needed = _memory[j] = (score_im1_1, p_ij_1, s_im1_1, y_im1_1, ye_im1_1, j)

            tmp.append(_needed)

            for jj in range(j + 1, len_prevb):

                if _memory[jj]:
                    _needed = _memory[jj]
                    score_im1_2, p_ij_2, s_im1_2, y_im1_2, ye_im1_2, njj = _needed
                    assert(jj == njj)
                else:
                    score_im1_2, p_ij_2, s_im1_2, _, y_im1_2, _ = prevb[jj]
                    ye_im1_2 = ye_im1[jj]
                    _needed = _memory[jj] = (score_im1_2, p_ij_2, s_im1_2, y_im1_2, ye_im1_2, jj)

                ifmerge = False
                if wargs.merge_way == 'Y': ifmerge = (y_im1_2 == y_im1_1)
                #if wargs.merge_way == 'Y':
                #    d = self.pdist(ye_im1_1.data.unsqueeze(0), ye_im1_2.data.unsqueeze(0))[0][0]
                    #d = cor_coef(y_im1_e1.data, y_im1_e2.data)
                    #if y_im1_1 == y_im1_2: print d
                    #if d < 1.: print d
                #    ifmerge = (d < 1000.)

                if ifmerge:
                    tmp.append(_needed)
                    used.append(jj)

            eq_classes[key] = tmp
            key += 1

    ##################################################################

    # NOTE: (Wen Zhang) create cube by sort row dimension

    ##################################################################

    #@exeTime
    def create_cube(self, bidx, eq_classes):
        # eq_classes: 0: [(score_im1, s_im1, y_im1, bp), ... ], 1:
        cube = []
        cnt_transed = len(self.hyps[0])

        # for each equivalence class
        for sub_cube_id, leq_class in eq_classes.iteritems():

            sub_cube_rowsz = len(leq_class)
            # (score_im1_1, p_ij_1, s_im1_1, y_im1_1, ye_im1_1, j)
            _ , _, _s_im1_r0, _, ye_im1_r0, _ = leq_class[0]
            '''
            _s_im1_r0, ye_im1_r0 = [], []
            for x in leq_class:
                _s_im1_r0.append(x[1])
                ye_im1_r0.append(x[-2])
            _s_im1_r0 = tc.mean(tc.stack(_s_im1_r0, dim=0), dim=0)
            ye_im1_r0 = tc.mean(tc.stack(ye_im1_r0, dim=0), dim=0)
            '''

            sub_cube = []
            _si, _ai, _cei = None, None, None

            # TODO sort the row dimension by average scores
            #if not sub_cube_rowsz == 1:
            #    _s_im1 = [tup[1] for tup in leq_class]
            #    _s_im1 = tc.mean(tc.stack(_s_im1, dim=0), dim=0)

            #t1 = time.time()
            _ai, _si, ye_im1, _pi, _, _, _ = self.decoder.step(_s_im1_r0, self.enc_src0, self.uh0, ye_im1_r0)
            self.C[4] += 1
            _logit = self.decoder.step_out(_si, ye_im1, _ai)
            self.C[5] += 1
            #t2 = time.time()

            _cei = self.classifier(_logit)
            #if wargs.vocab_norm:
                #_cei = self.classifier.pred_map(_logit)
                #t3 = time.time()
                #_cei = -self.classifier.log_prob(_cei)[-1]
            #else: _cei = -self.classifier.pred_map(_logit)
            #t4 = time.time()
            #total = t4 - t1
            #if bidx == 1 and sub_cube_id == 0:
            #wlog('Step:{:4.1%}, W-matrix:{:4.1%}, Softmax:{:4.1%}'.format(
            #    (t2 - t1)/total, (t3 - t2)/total, (t4 - t3)/total))
            _cei = _cei.cpu().data.numpy().flatten()    # (1,vocsize) -> (vocsize,)

            #next_krank_ids = part_sort(_cei, self.k - cnt_transed)
            next_krank_ids = part_sort(_cei, self.k)
            row_ksorted_ces = _cei[next_krank_ids]

            # add cnt for error The truth value of an array with more than one element is ambiguous
            for i, tup in enumerate(leq_class):
                sub_cube.append([
                    tup + (_ai, _si, _pi, row_ksorted_ces[j], wid, i, j, sub_cube_id, sub_cube_rowsz)
                    for j, wid in enumerate(next_krank_ids)])

            cube.append(sub_cube)

        # print created cube before generating current beam for debug ...
        '''
        debug('\n************************************************')
        n_sub_cube = len(cube)
        for sub_cube_id in xrange(n_sub_cube):
            sub_cube = cube[sub_cube_id]
            n_rows = len(sub_cube)
            debug('Group: {}, {} rows:'.format(sub_cube_id, n_rows))
            for rid in xrange(n_rows):
                report = ''
                sub_cube_line = sub_cube[rid]
                score_im1 = sub_cube_line[0][0]
                report += '{:6.4f} => '.format(score_im1)
                report_costs, report_ys = [], []
                for item in sub_cube_line:
                    c, y = item[-6], item[-5]
                    report_costs.append('{:6.4f}'.format(c))
                    report_ys.append('{}={}'.format(y, self.tvcb_i2w[y]))
                report += ('|'.join(report_costs) + ' => ' + '|'.join(report_ys))
                debug(report)
        debug('************************************************\n')
        '''

        return cube

    ##################################################################

    # NOTE: (Wen Zhang) Given cube, we calculate true score,
    # computation-expensive here

    ##################################################################

    def Push_heap(self, heap, bidx, citem):

        score_im1, _, s_im1, y_im1, ye_im1, bp, _ai, _si, _pi, _ce_jth, yi, \
                iexp, jexp, which, rsz = citem

        '''
        if rsz == 1 or iexp == 0:
            true_si = _si
            true_sci = score_im1 + _ce_jth
        else:
            if self.buf_state_merge[which][iexp]: true_si, _cei = self.buf_state_merge[which][iexp]
            else:
                a_i, true_si, ye_im1 = self.decoder.step(s_im1, self.enc_src0, self.uh0, ye_im1)
                self.C[4] += 1
                logit = self.decoder.step_out(true_si, ye_im1, a_i)
                self.C[5] += 1

                if wargs.vocab_norm: _cei = self.classifier(logit)
                else: _cei = -self.classifier.pred_map(logit)
                _cei = _cei.cpu().data.numpy().flatten()    # (1,vocsize) -> (vocsize,)

                self.buf_state_merge[which][iexp] = (true_si, _cei)

            true_sci = score_im1 + _cei[yi]
            #debug('| {:6.3f}={:6.3f}+{:6.3f}'.format(true_sci, score_im1, _cei[yi]))

        heapq.heappush(heap, (true_sci, score_im1, next(self.cnt), bp, true_si, yi, iexp, jexp, which))
        '''

        _sci = score_im1 + _ce_jth
        heapq.heappush(heap, (_sci, score_im1, next(self.cnt), s_im1,
                              ye_im1, bp, _si, _pi, yi, iexp, jexp, which, rsz))

    ##################################################################

    # NOTE: (Wen Zhang) cube pruning

    ##################################################################

    def cube_prune(self, bidx, cube):
        # search in cube (matrix(mergings) or vector(no mergings))
        n_sub_cube = len(cube)
        each_subcube_colsz, each_subcube_rowsz = [], []
        counter = 0
        extheap, self.buf_state_merge = [], []
        cnt_bp = (bidx >= 2)
        self.C[0] += n_sub_cube # count of total sub cubes
        for sub_cube_id in xrange(n_sub_cube):
            sub_cube = cube[sub_cube_id]
            rowsz = len(sub_cube)
            each_subcube_rowsz.append(rowsz)
            each_subcube_colsz.append(len(sub_cube[0]))
            self.C[1] += rowsz  # count of items in previous beam
            if cnt_bp: self.C[2] += rowsz
            # initial heap, starting from the left-top corner (best word) of each subcube
            # real score here ... may adding language model here ...
            # we should calculate the real score in current beam when pushing into heap
            self.Push_heap(extheap, bidx, sub_cube[0][0])
            #heapq.heappush(extheap, sub_cube[0][0])
            self.buf_state_merge.append([None] * rowsz)

        cnt_transed = len(self.hyps[0])
        while len(extheap) > 0 and counter < self.k - cnt_transed:
        #while len(extheap) > 0 and counter < self.k:

            #true_sci, score_im1, _, bp, true_si, pi, yi, iexp, jexp, which = heapq.heappop(extheap)

            _sci, score_im1, _, s_im1, ye_im1, bp, _si, _pi, yi, iexp, jexp, which, rsz = \
                    heapq.heappop(extheap)
            true_pi = _pi
            true_si = _si
            true_sci = _sci

            if rsz == 1 or iexp == 0:
                true_pi = _pi
                true_si = _si
                true_sci = _sci
            else:
                if self.buf_state_merge[which][iexp]: true_si, true_pi, _cei = self.buf_state_merge[which][iexp]
                else:
                    a_i, true_si, ye_im1, true_pi, _, _, _ = self.decoder.step(s_im1, self.enc_src0, self.uh0, ye_im1)
                    self.C[4] += 1
                    logit = self.decoder.step_out(true_si, ye_im1, a_i)
                    self.C[5] += 1

                    _cei = self.classifier(logit)
                    #else: _cei = -self.classifier.pred_map(logit)
                    _cei = _cei.cpu().data.numpy().flatten()    # (1,vocsize) -> (vocsize,)

                    self.buf_state_merge[which][iexp] = (true_si, true_pi, _cei)

                true_sci = score_im1 + _cei[yi]
                #debug('| {:6.3f}={:6.3f}+{:6.3f}'.format(true_sci, score_im1, _cei[yi]))

            if cnt_bp: self.C[3] += (bp + 1)
            if yi == EOS:
                # beam items count decrease 1
                if wargs.len_norm == 0: self.hyps[0].append((true_sci, true_sci, 0, yi, bp, bidx))
                elif wargs.len_norm == 1:
                    self.hyps[0].append(((true_sci / bidx), true_sci, 0, yi, bp, bidx))
                elif wargs.len_norm == 2:
                    lp, cp = self.lp_cp(bp, bidx)
                    #print true_sci, lp, cp, (true_sci/lp - cp)
                    self.hyps[0].append(((true_sci/lp - cp), true_sci, 0, yi, bp, bidx))
                debug('Gen hypo {}'.format(self.hyps[0][-1]))
                # last beam created and finish cube pruning
                if len(self.hyps[0]) == self.k: return True
            # generate one item in current beam
            else: self.next_beam_cur_sent.append((true_sci, true_pi, true_si, 0, yi, bp))
            self.beam[bidx] = [ self.next_beam_cur_sent ]
            self.preb_alpha.append(true_pi.squeeze(-1))

            whichsubcub = cube[which]
            # make sure we do not add repeadedly
            if jexp + 1 < each_subcube_colsz[which]:
                right = whichsubcub[iexp][jexp + 1]
                self.Push_heap(extheap, bidx, right)
                #heapq.heappush(extheap, right)
            if iexp + 1 < each_subcube_rowsz[which]:
                down = whichsubcub[iexp + 1][jexp]
                self.Push_heap(extheap, bidx, down)
                #heapq.heappush(extheap, right)
            counter += 1
        if self.attent_probs is not None: self.attent_probs[0].append(tc.stack(self.preb_alpha, 1))
        return False

    def lp_cp(self, bp, bidx):

        ys_pi = []
        for i in reversed(xrange(1, bidx)):
            _, p_im1, _, w, bp = self.beam[i][0][bp]
            ys_pi.append(p_im1)
        if len(ys_pi) == 0: return 1.0, 0.0
        ys_pi = tc.stack(ys_pi, dim=0).sum(0)   # (slen, 1)
        m = (ys_pi > 1.0).float()
        ys_pi = ys_pi * (1. - m) + m
        lp = ((5+bidx-1) ** wargs.length_norm) / ((5+1)**wargs.length_norm)
        cp = wargs.cover_penalty * (ys_pi.log().sum().data[0])

        return lp, cp

    def cube_pruning(self):

        self.true_bidx = [0]

        for bidx in range(1, self.maxL + 1):

            self.preb_alpha, self.next_beam_cur_sent = [], []
            eq_classes = OrderedDict()
            self.merge(bidx, eq_classes)

            # create cube and generate next beam from cube
            cube = self.create_cube(bidx, eq_classes)

            if self.cube_prune(bidx, cube):

                debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                sorted_hyps = sorted(self.hyps[0], key=lambda tup: tup[0])
                for hyp in sorted_hyps: debug('{}'.format(hyp))
                best_hyp = sorted_hyps[0]
                debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))
                self.batch_tran_cands[0] = [back_tracking(self.beam, 0, hyp, \
                    self.attent_probs[0] if self.attent_probs is not None \
                                            else None) for hyp in sorted_hyps]
                return
                #return back_tracking(self.beam, best_hyp)

            self.beam[bidx] = sorted(self.beam[bidx], key=lambda tup: tup[0])
            debug('beam {} ----------------------------'.format(bidx))
            for b in self.beam[bidx][0]: debug(b[0:1] + b[-3:])    # do not output state
            # because of the the estimation of P(f|abcd) as P(f|cd), so the generated beam by
            # cube pruning may out of order by loss, so we need to sort it again here
            # losss from low to high

        # no early stop, back tracking
        n_remainings = len(self.beam[self.maxL])   # loop ends, how many sentences left
        self.no_early_best(n_remainings)

        '''
        # no early stop, back tracking
        if len(self.hyps[0]) == 0:
            #debug('No early stop, no hyp ending with EOS, select one length {} '.format(self.maxL))
            best_hyp = self.beam[self.maxL][0]
            if wargs.len_norm == 0:
                best_hyp = (best_hyp[0], ) + best_hyp[-2:] + (self.maxL, )
            elif wargs.len_norm == 1:
                best_hyp = (best_hyp[0]/self.maxL, best_hyp[0], ) + best_hyp[-2:] + (self.maxL, )
            elif wargs.len_norm == 2:
                lp, cp = self.lp_cp(0, self.maxL+1)
                best_hyp = (best_hyp[0]/lp - cp, best_hyp[0], ) + best_hyp[-2:] + (self.maxL, )

        else:
            #debug('No early stop, no enough {} hyps ending with EOS, select the best '
            #      'one from {} hyps.'.format(self.k, len(self.hyps)))
            sorted_hyps = sorted(self.hyps[0], key=lambda tup: tup[0])
            #for hyp in sorted_hyps: debug('{}'.format(hyp))
            best_hyp = sorted_hyps[0]

        #debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))
        return back_tracking(self.beam, best_hyp)
        '''

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


