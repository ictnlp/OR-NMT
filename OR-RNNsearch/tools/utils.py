from __future__ import division

import sys
import os
import re
import numpy
import shutil
import json
import subprocess
import math
import random

import torch as tc
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.append('../')
import wargs

def str1(content, encoding='utf-8'):
    return json.dumps(content, encoding=encoding, ensure_ascii=False, indent=4)
    pass

#DEBUG = True
DEBUG = False
MAX_SEQ_SIZE = 5000
PAD_WORD = '<pad>'
UNK_WORD = 'unk'
BOS_WORD = '<b>'
EOS_WORD = '<e>'
RESERVED_TOKENS = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD = RESERVED_TOKENS.index(PAD_WORD)  # 0
UNK = RESERVED_TOKENS.index(UNK_WORD)  # 1
BOS = RESERVED_TOKENS.index(BOS_WORD)  # 2
EOS = RESERVED_TOKENS.index(EOS_WORD)  # 3

def load_model(model_path):
    wlog('Loading pre-trained model ... from {} '.format(model_path), 0)
    state_dict = tc.load(model_path, map_location=lambda storage, loc: storage)
    if len(state_dict) == 4:
        model_dict, eid, bid, optim = state_dict['model'], state_dict['epoch'], state_dict['batch'], state_dict['optim']
        rst = ( model_dict, eid, bid, optim )
    elif len(state_dict) == 5:
        model_dict, class_dict, eid, bid, optim = state_dict['model'], state_dict['class'], state_dict['epoch'], state_dict['batch'], state_dict['optim']
        rst = ( model_dict, class_dict, eid, bid, optim )
    wlog('at epoch {} and batch {}'.format(eid, bid))
    wlog(optim)
    return rst

def toVar(x, isCuda=None, volatile=False):

    if not isinstance(x, tc.autograd.variable.Variable):
        if isinstance(x, int): x = tc.Tensor([x])
        elif isinstance(x, list): x = tc.Tensor(x)
        x = Variable(x, requires_grad=False, volatile=volatile)
        if isCuda is not None: x = x.cuda()

    return x

def clip(x, rate=0.):

    b1 = (x < 1 - rate).float()
    b2 = (x > 1 + rate).float()
    b3 = (((1 - rate <= x) + (x <= 1 + rate)) > 1).float()

    return b1 * (1 - rate) + b2 * (1 + rate) + b3 * x

def rm_elems_byid(l, ids):

    isTensor = isinstance(l, tc.FloatTensor)
    isTorchVar = isinstance(l, tc.autograd.variable.Variable)
    if isTensor is True: l = l.transpose(0, 1).tolist()
    if isTorchVar is True: l = l.transpose(0, 1).data.tolist() #  -> (B, srcL)

    if isinstance(ids, int): del l[ids]
    elif len(ids) == 1: del l[ids[0]]
    else:
        for idx in ids: l[idx] = PAD_WORD
        l = filter(lambda a: a != PAD_WORD, l)

    if isTensor is True: l = tc.Tensor(l).transpose(0, 1)  # -> (srcL, B')
    if isTorchVar is True:
        l = Variable(tc.Tensor(l).transpose(0, 1), requires_grad=False, volatile=True)
        if wargs.gpu_id: l = l.cuda()

    return l

# x, y are torch Tensors
def cor_coef(a, b, eps=1e-20):

    E_a, E_b = tc.mean(a), tc.mean(b)
    E_a_2, E_b_2 = tc.mean(a * a), tc.mean(b * b)
    rl_rho = tc.mean(a * b) - E_a * E_b
    #print 'a',rl_rho.data[0]
    D_a, D_b = E_a_2 - E_a * E_a, E_b_2 - E_b * E_b

    rl_rho = rl_rho / ( tc.sqrt(D_a * D_b) + eps )  # number stable
    del E_a, E_b, E_a_2, E_b_2, D_a, D_b

    return rl_rho

def format_time(time):
    '''
        :type time: float
        :param time: the number of seconds

        :print the text format of time
    '''
    rst = ''
    if time < 0.1: rst = '{:7.2f} ms'.format(time * 1000)
    elif time < 60: rst = '{:7.5f} sec'.format(time)
    elif time < 3600: rst = '{:6.4f} min'.format(time / 60.)
    else: rst = '{:6.4f} hr'.format(time / 3600.)

    return rst

def append_file(filename, content):

    f = open(filename, 'a')
    f.write(content + '\n')
    f.close()

def str_cat(pp, name):

    return '{}_{}'.format(pp, name)

def wlog(obj, newline=1):

    if newline == 1: sys.stderr.write('{}\n'.format(obj))
    else: sys.stderr.write('{}'.format(obj))
    #if newline == 1: print(obj, file=sys.stderr, flush=True)
    #else: print(obj, file=sys.stderr, end='', flush=True)
    sys.stderr.flush()

def debug(s, newline=1):

    if DEBUG is True:
        sys.stderr.flush()
        if newline == 1: sys.stderr.write('{}\n'.format(s))
        else: sys.stderr.write('{}'.format(s))
        #if newline == 1: print('{}\n'.format(s))
        #else: print(s)
        sys.stderr.flush()

def get_gumbel(LB, V, eps=1e-30):

    return Variable(
        -tc.log(-tc.log(tc.Tensor(LB, V).uniform_(0, 1) + eps) + eps), requires_grad=False)

def BLToStrList(x, xs_L, return_list=False):

    x = x.data.tolist()
    B, xs = len(x), []
    for bidx in range(B):
        x_one = numpy.asarray(x[bidx][:int(xs_L[bidx])])
        #x_one = str(x_one.astype('S10'))[1:-1].replace('\n', '')
        x_one = str(x_one.astype('S10')).replace('\n', '')
        #x_one = x_one.__str__().replace('  ', ' ')[2:-1]
        xs.append(x_one)
    return xs if return_list is True else '\n'.join(xs)

def init_params(p, name='what', init_D='U', a=0.01):

    if 'layer_norm' in name: return     # parameters of layer norm have been initialized
    p_dim = p.dim()
    if init_D == 'U':           # uniform distribution for all parameters
        p.data.uniform_(-a, a)
        wlog('{:7}-{} -> grad {}\t{}'.format('Uniform', a, p.requires_grad, name))
    elif init_D == 'X':         # xavier distribution for 2-d parameters
        if p_dim == 1 or (p_dim == 2 and (p.size(0) == 1 or p.size(1) == 1)):
            p.data.zero_()
            wlog('{:7} -> grad {}\t{}'.format('Zero', p.requires_grad, name))
        else:
            init.xavier_uniform_(p)
            wlog('{:7} -> grad {}\t{}'.format('Xavier', p.requires_grad, name))
    elif init_D == 'N':         # normal distribution for 2-d parameters
        if p_dim == 1 or (p_dim == 2 and (p.size(0) == 1 or p.size(1) == 1)):
            p.data.zero_()
            wlog('{:7} -> grad {}\t{}'.format('Zero', p.requires_grad, name))
        else:
            p.data.normal_(0, 0.01)
            wlog('{:7} -> grad {}\t{}'.format('Normal', p.requires_grad, name))

def init_dir(dir_name, delete=False):

    if not dir_name == '':
        if os.path.exists(dir_name):
            if delete:
                shutil.rmtree(dir_name)
                wlog('\n{} exists, delete'.format(dir_name))
            else:
                wlog('\n{} exists, no delete'.format(dir_name))
        else:
            os.mkdir(dir_name)
            wlog('\nCreate {}'.format(dir_name))

def part_sort(vec, num):
    '''
    vec:    [ 3,  4,  5, 12,  1,  3,  29999, 33,  2, 11,  0]
    '''

    idx = numpy.argpartition(vec, num)[:num]

    '''
    put k-min numbers before the _th position and get indexes of the k-min numbers in vec (unsorted)
    k-min vals:    [ 1,  0,  2, 3,  3,  ...]
    idx = np.argpartition(vec, 5)[:4]:
        [ 4, 10,  8,  0,  5]
    '''

    kmin_vals = vec[idx]

    '''
    kmin_vals:  [1, 0, 2, 3, 3]
    '''

    k_rank_ids = numpy.argsort(kmin_vals)

    '''
    k_rank_ids:    [1, 0, 2, 3, 4]
    '''

    k_rank_ids_invec = idx[k_rank_ids]

    '''
    k_rank_ids_invec:  [10,  4,  8,  0,  5]
    '''

    '''
    sorted_kmin = vec[k_rank_ids_invec]
    sorted_kmin:    [0, 1, 2, 3, 3]
    '''

    return k_rank_ids_invec

# beam search
def init_beam(beam, s0=None, cnt=50, score_0=0.0, loss_0=0.0, dyn_dec_tup=None, cp=False, transformer=False):
    del beam[:]
    for i in range(cnt + 1):
        ibeam = []  # one beam [] for one char besides start beam
        beam.append(ibeam)

    if cp is True:
        beam[0] = [ [ (loss_0, None, s0, 0, BOS, 0) ] ]
        return
    # indicator for the first target word (<b>)
    if dyn_dec_tup is not None:
        beam[0].append((loss_0, dyn_dec_tup, s0, BOS, 0))
    elif wargs.len_norm == 2:
        beam[0] = [[ (loss_0, None, s0[i], i, BOS, 0) ] for i in range(s0.size(0))]
    #elif with_batch == 0:
    #    beam[0].append((loss_0, s0, BOS, 0))
    else:
        beam[0] = [[ (loss_0, s0[i], i, BOS, 0) ] for i in range(s0.size(0))]

def back_tracking(beam, bidx, best_sample_endswith_eos, attent_probs=None):
    # (0.76025655120611191, [29999], 0, 7)
    best_loss, accum, _, w, bp, endi = best_sample_endswith_eos
    # starting from bp^{th} item in previous {end-1}_{th} beam of eos beam, w is <eos>
    seq = []
    attent_matrix = [] if attent_probs is not None else None
    check = (len(beam[0][0][0]) == 5)
    #print len(attent_probs), endi
    for i in reversed(range(0, endi)): # [0, endi-1], with <bos> 0 and no <eos> endi==self.maxL
        # the best (minimal sum) loss which is the first one in the last beam,
        # then use the back pointer to find the best path backward
        # <eos> is in pos endi, we do not keep <eos>
        if check is True:
            _, _, true_bidx, w, backptr = beam[i][bidx][bp]
            #if isinstance(true_bidx, int): assert true_bidx == bidx
        else: _, _, _, _, w, backptr = beam[i][bidx][bp]
        seq.append(w)
        bp = backptr
        # ([first word, ..., last word]) with bos and no eos, bos haven't align
        if attent_matrix is not None and i != 0:
            attent_matrix.append(attent_probs[i-1][:, bp])

    if attent_probs is not None and len(attent_matrix) > 0:
        # attent_matrix: (trgL, srcL)
        attent_matrix = tc.stack(attent_matrix[::-1], dim=0)
        attent_matrix = attent_matrix.cpu().data.numpy()

    # seq (bos, t1, t2, t3, t4, ---)
    # att (---, a0, a1, a2, a3, a4 ) 
    return seq[::-1], best_loss, attent_matrix # reverse

def filter_reidx(best_trans, tV_i2w=None, attent_matrix=None, ifmv=False, ptv=None):

    if ifmv and ptv is not None:
        # OrderedDict([(0, 0), (1, 1), (3, 5), (8, 2), (10, 3), (100, 4)])
        # reverse: OrderedDict([(0, 0), (1, 1), (5, 3), (2, 8), (3, 10), (4, 100)])
        # part[index] get the real index in large target vocab firstly
        true_idx = [ptv[i] for i in best_trans]
    else:
        true_idx = best_trans

    remove_ids, filter_ids = [], []
    for idx in range(1, len(true_idx)):
        widx = true_idx[idx]
        if widx == BOS or widx == EOS or widx == PAD:
            remove_ids.append(idx - 1)
        else:
            filter_ids.append(widx)
    #true_idx = filter(lambda y: y != BOS and y != EOS, true_idx)

    return idx2sent(filter_ids, tV_i2w), filter_ids, idx2sent(true_idx, tV_i2w), \
            true_idx, numpy.delete(attent_matrix, remove_ids, 0)

def sent_filter(sent):

    list_filter = filter(lambda x: x != PAD and x!= BOS and x != EOS, sent)

    return list(list_filter)

def idx2sent(vec, vcb_i2w):
    # vec: [int, int, ...]
    if isinstance(vcb_i2w, dict):
        r = [vcb_i2w[idx] for idx in vec]
        sent = ' '.join(r)
    else:
        sent = vcb_i2w.decode(vec)
    return sent

def dec_conf():

    wlog('\n######################### Construct Decoder #########################\n')
    if wargs.search_mode == 0: wlog('# Greedy search => ')
    elif wargs.search_mode == 1: wlog('# Naive beam search => ')
    elif wargs.search_mode == 2: wlog('# Cube pruning => ')

    wlog('\t Beam size: {}'
         '\n\t KL_threshold: {}'
         '\n\t Batch decoding: {}'
         '\n\t Vocab normalized: {}'
         '\n\t Length normalized: {}'
         '\n\t Manipulate vocab: {}'
         '\n\t Cube pruning merge way: {}'
         '\n\t Average attent: {}\n\n'.format(
             wargs.beam_size,
             wargs.m_threshold,
             True if wargs.with_batch else False,
             True if wargs.vocab_norm else False,
             wargs.len_norm,
             True if wargs.with_mv else False,
             wargs.merge_way,
             True if wargs.avg_att else False
         )
    )

def print_attention_text(attention_matrix, source_tokens, target_tokens, threshold=0.9, isP=False):
    """
    Return the alignment string from the attention matrix.
    Prints the attention matrix to standard out.
    :param attention_matrix: The attention matrix, np.ndarray, (trgL, srcL)
    :param source_tokens: A list of source tokens, List[str]
    :param target_tokens: A list of target tokens, List[str]
    :param threshold: The threshold for including an alignment link in the result, float
    """
    if not attention_matrix.shape[0] == len(target_tokens):
        wlog(attention_matrix)
        wlog('-------------------------')
        wlog(target_tokens)

    if len(target_tokens) == 0: return ''

    assert attention_matrix.shape[0] == len(target_tokens), \
            'attention shape: ' + str(attention_matrix.shape) + ', target: ' + str(len(target_tokens))

    if isP is True:
        wlog('  ', 0)
        for j in target_tokens: wlog('---', 0)
        wlog('')

    alnList = []
    src_max_ids, src_max_p = attention_matrix.argmax(1) + 1, attention_matrix.max(1)
    for (i, f_i) in enumerate(source_tokens):
        #maxJ, maxP = 0, 0.0

        if isP is True: wlog(' |', 0)
        for (j, _) in enumerate(target_tokens):
            align_prob = attention_matrix[j, i]
            if i == 0:  # start from 1
                alnList.append('{}:{}/{:.2f}'.format(src_max_ids[j], j+1, src_max_p[j]))
                #if maxP >= 0.5:
                #    alnList.append('{}:{}/{:.2f}'.format(i + 1, maxJ + 1, maxP))    # start from 1 here
            if isP is True:
                if align_prob > threshold: wlog('(*)', 0)
                elif align_prob > 0.4: wlog('(?)', 0)
                else: wlog('   ', 0)
            #if align_prob > maxP: maxJ, maxP = j, align_prob

        if isP is True: wlog(' | {}'.format(f_i))

    if isP is True:
        wlog('  ', 0)
        for j in target_tokens: wlog('---', 0)
        wlog('')
        for k in range(max(map(len, target_tokens))):
            wlog('  ', 0)
            for word in target_tokens:
                letter = word[k] if len(word) > k else ' '
                wlog(' {} '.format(letter), 0)
            wlog('')
        wlog('')

    return ' '.join(alnList)

def plot_attention(attention_matrix, source_tokens, target_tokens, filename):
    """
    Uses matplotlib for creating a visualization of the attention matrix.
    :param attention_matrix: The attention matrix, np.ndarray
    :param source_tokens: A list of source tokens, List[str]
    :param target_tokens: A list of target tokens, List[str]
    :param filename: The file to which the attention visualization will be written to, str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pylab import mpl

    #matplotlib.rc('font', family='sans-serif')
    #matplotlib.rc('font', serif='HelveticaNeue')
    #matplotlib.rc('font', serif='SimHei')
    #matplotlib.rc('font', serif='Microsoft YaHei')
    #mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #mpl.rcParams['font.sans-serif'] = ['SimHei']
    #plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #mpl.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    #plt.rcParams['axes.unicode_minus'] = False
    #mpl.rcParams['axes.unicode_minus'] = False
    #zh_font = mpl.font_manager.FontProperties(fname='/home5/wen/miniconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')
    zh_font = mpl.font_manager.FontProperties(
        fname='/home/wen/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Microsoft Yahei.ttf')
    en_font = mpl.font_manager.FontProperties(
        fname='/home/wen/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Microsoft Yahei.ttf')
        #fname='/home/wen/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')

    assert attention_matrix.shape[0] == len(target_tokens)

    plt.clf()
    #plt.imshow(attention_matrix.transpose(), interpolation="nearest", cmap="Greys")
    plt.imshow(attention_matrix, interpolation="nearest", cmap="Greys")
    #plt.xlabel("Source", fontsize=16)
    #plt.ylabel("Target", fontsize=16)

    #plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('top')
    #plt.xticks(fontsize=18, fontweight='bold')
    plt.xticks(fontsize=20)
    #plt.yticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=20)

    #plt.grid(True, which='minor', linestyle='-')
    #plt.gca().set_xticks([i for i in range(0, len(target_tokens))])
    #plt.gca().set_yticks([i for i in range(0, len(source_tokens))])
    plt.gca().set_xticks([i for i in range(0, len(source_tokens))])
    plt.gca().set_yticks([i for i in range(0, len(target_tokens))])
    #plt.gca().set_xticklabels(source_tokens, rotation='vertical')
    plt.gca().set_xticklabels(source_tokens, rotation=70, fontweight='bold', FontProperties=zh_font)
    plt.gca().tick_params(axis='x', labelsize=20)
    #plt.gca().set_xticklabels(source_tokens, rotation=70, fontsize=20)

    #source_tokens = [unicode(k, "utf-8") for k in source_tokens]
    #plt.gca().set_yticklabels(source_tokens, rotation='horizontal', fontproperties=zh_font)
    #plt.gca().set_yticklabels(source_tokens, rotation='horizontal')
    plt.gca().set_yticklabels(target_tokens, fontsize=24, fontweight='bold', FontProperties=en_font)

    plt.tight_layout()
    #plt.draw()
    #plt.show()
    #plt.savefig(filename, format='png', dpi=400)
    #plt.grid(True)
    #plt.savefig(filename, dpi=400)
    plt.savefig(filename, format='svg', dpi=600, bbox_inches='tight')
    #plt.savefig(filename)
    wlog("Saved alignment visualization to " + filename)

def ids2Tensor(list_wids, bos_id=None, eos_id=None):
    # input: list of int for one sentence
    list_idx = [bos_id] if bos_id else []
    for wid in list_wids: list_idx.extend([wid])
    list_idx.extend([eos_id] if eos_id else [])
    return tc.LongTensor(list_idx)

def lp_cp(bp, beam_idx, bidx, beam):
    assert len(beam[0][0][0]) == 6, 'require attention prob for alpha'
    cp = 0.
    if wargs.beta_cover_penalty > 0.:
        ys_pi = []
        for i in reversed(range(1, beam_idx)):
            _, p_im1, _, _, w, bp = beam[i][bidx][bp]
            ys_pi.append(p_im1)
        if len(ys_pi) == 0: return 1.0, 0.0
        ys_pi = tc.stack(ys_pi, dim=0).sum(0)   # (part_trg_len, src_len) -> (src_len, )
        #m = ( ys_pi > 1.0 ).float()
        #ys_pi = ys_pi * ( 1. - m ) + m
        penalty = tc.min(ys_pi, ys_pi.clone().fill_(1.0)).log().sum().item()
        cp = wargs.beta_cover_penalty * penalty
    #lp = ( ( 5 + beam_idx - 1 ) ** wargs.alpha_len_norm ) / ( (5 + 1) ** wargs.alpha_len_norm )
    lp = ( ( 5 + beam_idx + 1 ) ** wargs.alpha_len_norm ) / ( (5 + 1) ** wargs.alpha_len_norm )

    return lp, cp

'''
    a: add previous input tensor
    n: apply normalization
    d: apply dropout
  For example, if sequence=="dna", output is normalize(dropout(x))+previous_value
  Args:
    pre_layer_in: Tensor to be added as a residual connection ('a')
    pre_layer_out: Tensor to be transformed.
    handle_type: 'dna'
    normlizer: None, Layer_Norm, nn.BatchNorm1d, 'noam'
    epsilon: a float (parameter for normalization)
    dropout_rate: a float
  Returns:
    a Tensor
'''
def layer_prepostprocess(pre_layer_out, pre_layer_in=None, handle_type=None, normlizer=None,
                         epsilon=1e-6, dropout_rate=0., training=True):
    if handle_type is None: return pre_layer_out
    if 'a' in handle_type: assert pre_layer_in is not None, 'Residual requires previous input !'
    for c in handle_type:
      if c == 'a': pre_layer_out += pre_layer_in
      elif c == 'n':
        if normlizer == 'noam': # One version of layer normalization
            pre_layer_out = F.normalize(pre_layer_out, p=2, dim=-1, eps=1e-20)
        else: pre_layer_out = normlizer(pre_layer_out)
      elif c == 'd': pre_layer_out = F.dropout(pre_layer_out, p=dropout_rate, training=training)
      else: wlog('Unknown handle type {}'.format(c))
    return pre_layer_out

def schedule_sample_word(_h, _g, ss_eps, y_tm1_gold, y_tm1_hypo):

    if y_tm1_hypo is None: return y_tm1_gold

    return y_tm1_hypo * _h + y_tm1_gold * _g

def schedule_sample(ss_eps, y_tm1_gold, y_tm1_hypo):

    if y_tm1_hypo is None: return y_tm1_gold

    return y_tm1_hypo if random.random() > ss_eps else y_tm1_gold

def schedule_bow_lambda(epo_idx, max_lambda=3.0, k=0.1, alpha=0.1):
    return min(max_lambda, k + alpha * epo_idx)

def ss_prob_decay(i):

    ss_type, k = wargs.ss_type, wargs.ss_k
    if ss_type == 1:    # Linear decay
        ss = wargs.ss_prob_begin - ( wargs.ss_decay_rate * i )
        if ss < wargs.ss_prob_end:
            prob_i = wargs.ss_prob_end
            wlog('[Linear] schedule sampling probability do not change {}'.format(prob_i))
        else:
            prob_i = ss
            wlog('[Linear] decay schedule sampling probability to {}'.format(prob_i))

    elif ss_type == 2:  # Exponential decay
        prob_i = numpy.power(k, i)
        wlog('[Exponential] decay schedule sampling probability to {}'.format(prob_i))

    elif ss_type == 3:  # Inverse sigmoid decay
        prob_i = k / ( k + numpy.exp( ( i / k ) ) )
        wlog('[Inverse] decay schedule sampling probability to {}'.format(prob_i))

    return prob_i

from tools.mteval_bleu import *
def batch_search_oracle(B_hypos_list, y_LB, y_mask_LB):

    #print B_hypos_list
    #y_Ls contains the <s> and <e> of each sentence in one batch
    y_maxL, y_Ls = y_mask_LB.size(0), y_mask_LB.sum(0).data.int().tolist()
    #print y_maxL, y_Ls
    #for bidx, hypos_list in enumerate(B_hypos_list):
    #    for hypo in hypos_list:
    #        hypo += [PAD] * (y_maxL - y_Ls[bidx])
    #print B_hypos_list
    # B_hypos_list: [[[w0, w1, w2, ..., ], [w0, w1]], [sent0, sent1], [...]]
    oracles = []
    B_ys_list = BLToStrList(y_LB.t(), [l-1 for l in y_Ls], True) # remove <s> and <e>
    for bidx, (hyps, gold) in enumerate(zip(B_hypos_list, B_ys_list)):
        oracle, max_bleu = hyps[0], 0.
        for h in hyps:
            h_ = str(numpy.array(h[1:]).astype('S10')).replace('\n', '')    # remove bos
            # do not consider <s> and <e> when calculating BLEU
            assert len(h_.split(' ')) == len(gold.split(' '))
            BLEU = bleu(h_, [gold], logfun=debug)
            if BLEU > max_bleu:
                max_bleu = BLEU
                oracle = h
        oracles.append(oracle + [EOS] + [PAD] * (y_maxL - y_Ls[bidx]))
        # same with y_LB
    oracles = Variable(tc.Tensor(oracles).long().t(), requires_grad=False)

    return oracles

def grad_checker(model, _checks=None):

    _grad_nan = False
    for n, p in model.named_parameters():
        if p.grad is None:
            debug('grad None | {}'.format(n))
            continue
        tmp_grad = p.grad.data.cpu().numpy()
        if numpy.isnan(tmp_grad).any(): # we check gradient here for vanishing Gradient
            wlog("grad contains 'nan' | {}".format(n))
            #wlog("gradient\n{}".format(tmp_grad))
            _grad_nan = True
        if n == 'decoder.l_f1_0.weight' or n == 's_init.weight' or n=='decoder.l_f1_1.weight' \
           or n == 'decoder.l_conv.0.weight' or n == 'decoder.l_f2.weight':
            debug('grad zeros |{:5} {}'.format(str(not numpy.any(tmp_grad)), n))

    if _grad_nan is True and _checks is not None:
        for _i, items in enumerate(_checks):
            wlog('step {} Variable----------------:'.format(_i))
            #for item in items: wlog(item.cpu().data.numpy())
            wlog('wen _check_tanh_sa ---------------')
            wlog(items[0].cpu().data.numpy())
            wlog('wen _check_a1_weight ---------------')
            wlog(items[1].cpu().data.numpy())
            wlog('wen _check_a1 ---------------')
            wlog(items[2].cpu().data.numpy())
            wlog('wen alpha_ij---------------')
            wlog(items[3].cpu().data.numpy())
            wlog('wen before_mask---------------')
            wlog(items[4].cpu().data.numpy())
            wlog('wen after_mask---------------')
            wlog(items[5].cpu().data.numpy())

def proc_bpe(input_fname, output_fname):

    fin = open(input_fname, 'r')
    contend = fin.read()
    fin.close()

    contend = re.sub('(@@ )|(@@ ?$)', '', contend)

    fout = open(output_fname, 'w')
    fout.write(contend)
    fout.close()

def proc_luong(input_fname, output_fname):

    fin = open(input_fname, 'r')
    contend = fin.read()
    fin.close()

    contend = re.sub('( ?##AT##-##AT## ?)', '', contend)

    fout = open(output_fname, 'w')
    fout.write(contend)
    fout.close()

def grab_all_trg_files(filename):

    file_names = []
    file_realpath = os.path.realpath(filename)
    data_dir = os.path.dirname(file_realpath)  # ./data
    file_prefix = os.path.basename(file_realpath)  # train.trg
    for fname in os.listdir(data_dir):
        if fname.startswith(file_prefix):
            file_path = os.path.join(data_dir, fname)
            #wlog('\t{}'.format(file_path))
            file_names.append(file_path)
    wlog('NOTE: Target side has {} references.'.format(len(file_names)))
    return file_names

def sort_batches(srcs, trgs, slens, batch_size, k=1):

    #assert len(trgFs) == 1, 'Unsupport to sort validation set in k batches.'
    final_srcs, final_trgs, train_size = [], [], len(srcs)
    if k == 0:
        wlog('sorting the whole dataset by ascending order of source length ... ', False)
        # sort the whole training data by ascending order of source length
        #_, sorted_idx = tc.sort(tc.IntTensor(slens))
        sorted_idx = sorted(range(train_size), key=lambda k: slens[k])
        final_srcs = [srcs[k] for k in sorted_idx]
        final_trgs = [trgs[k] for k in sorted_idx]
        wlog('done.')
    elif k > 1:
        wlog('sorting for each {} batches ... '.format(k), False)
        k_batch = batch_size * k
        number = int(math.ceil(train_size / k_batch))
        for start in range(number + 1):
            bsrcs = srcs[start * k_batch : (start + 1) * k_batch]
            btrgs = trgs[start * k_batch : (start + 1) * k_batch]
            bslens = slens[start * k_batch : (start + 1) * k_batch]
            #_, sorted_idx = tc.sort(tc.IntTensor(bslens))
            sorted_idx = sorted(range(len(bslens)), key=lambda k: bslens[k])
            final_srcs += [bsrcs[k] for k in sorted_idx]
            final_trgs += [btrgs[k] for k in sorted_idx]
        wlog('done.')
    if len(final_srcs) == 0 and len(final_trgs) == 0: return srcs, trgs
    else: return final_srcs, final_trgs



