from __future__ import division

import numpy
from tools.utils import *

##################################################################

# Wen Zhang: maximal sampling

##################################################################

class Greedy(object):

    def __init__(self, tvcb_i2w=None, ptv=None):

        self.lqc = [0] * 10
        self.tvcb_i2w = tvcb_i2w
        self.ptv = ptv

    def greedy_trans(src_sent, fs, switchs, trg_vocab_i2w, maxlen=40):
        counter = [0, 0, 0, 0, 0, 0, 0, 0]
        f_init, f_nh, f_na, f_ns, f_mo, f_ws, f_ps, f_p = fs
        ifvalid, ifbatch, ifscore, ifnorm, ifmv = switchs

        src_sent = src_sent[0] if ifvalid else src_sent  # numpy ndarray
        # subdict set [0,2,6,29999, 333]
        sub_trg_vocab_i2w = numpy.asarray(src_sent[1], dtype='int32') if ifvalid else None

        np_src_sent = numpy.asarray(src_sent, dtype='int64')
        if np_src_sent.ndim == 1:  # x (5,)
            # x(5, 1), (src_sent_len, batch_size)
            np_src_sent = np_src_sent[:, None]
        '''
    <type 'numpy.ndarray'> (7, 1)
    [[10811]
     [  140]
     [  217]
     [   19]
     [ 1047]
     [  482]
     [29999]]
        '''
        src_sent_len = np_src_sent.shape[0]
        maxlen = src_sent_len * 2     # x(src_sent_len, batch_size)
        eos_id = len(trg_vocab_i2w) - 1
        bos_id = 0

        s_im1, ctx0 = f_init(np_src_sent)   # np_src_sent (sl, 1), beam==1
        counter[0] += 1
        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        y_im1 = [-1]  # indicator for the first target word (bos <S>)

        best_trans = []
        best_model_loss = 0.0
        for i in xrange(maxlen + 1):
            # ctx = numpy.tile(ctx0, [1, 1])
            # (1, temb), (src_len, 1, src_nhids*2), (1, trg_nhids)
            hi = f_nh(*[y_im1, s_im1])
            counter[1] += 1
            pi, ai = f_na(*[ctx0, hi])
            counter[2] += 1
            s_im1 = f_ns(*[hi, ai])  # note, s_im1 should be updated!
            counter[3] += 1
            mo = f_mo(*[y_im1, ai, s_im1])
            counter[4] += 1
            next_scores = f_ws(*[mo])  # the larger the better
            counter[5] += 1
            if ifscore:
                if False:
                    next_loss = -numpy.log(sigmoid_better(next_scores))
                else:
                    next_loss = -next_scores
            else:
                next_loss = f_p(*[next_scores])
                counter[7] += 1
                # next_loss = -numpy.log(next_probs)
            next_loss_flat = next_loss.flatten()
            min_widx = numpy.argmin(next_loss_flat)
            best_trans.append(min_widx)
            minloss = next_loss_flat[min_widx]
            y_im1 = [min_widx]
            # positive, do not forget add score of eos
            best_model_loss += minloss
            if min_widx == eos_id:
                break

        norm_loss = (best_model_loss / len(best_trans)) if ifnorm else best_model_loss
        logger.info('@source length[{}], translation length[{}], maxlen[{}], loss'
                    '[{}]'.format(src_sent_len, len(best_trans), maxlen, norm_loss))
        logger.info('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] p[{}]'.format(*counter))
        return _filter_reidx(bos_id, eos_id, best_trans, trg_vocab_i2w, ifmv, sub_trg_vocab_i2w)




