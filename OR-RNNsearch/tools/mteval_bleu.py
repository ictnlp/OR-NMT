from __future__ import division

import os
import math
import re
import sys
import numpy
import string
from zhon import hanzi

def wlog(obj, newline=1):

    if newline == 1: sys.stderr.write('{}\n'.format(obj))
    else: sys.stderr.write('{}'.format(obj))

def zh_to_chars(s):

    regex = []

    # Match a whole word:
    regex += [r'[A-Za-z]+']

    # Match a single CJK character:
    regex += [r'[\u4e00-\ufaff]']

    # Match one of anything else, except for spaces:
    #regex += [r'^\s']

    # Match the float
    regex += [r'[-+]?\d*\.\d+|\d+']

    # Match chinese float
    ch_punc = hanzi.punctuation
    regex += [r'[{}]'.format(ch_punc)]	# point .

    # Match the punctuation
    regex += [r'[.]+']	# point .

    punc = string.punctuation
    punc = punc.replace('.', '')
    regex += [r'[{}]'.format(punc)]

    regex = '|'.join(regex)
    r = re.compile(regex)

    return r.findall(s)

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

def length_bleu(src_fpath, ref_fpaths, trans_fpath, ngram=4, cased=False, char=False,
                min_len=0, max_len=80, len_interval=10):

    wlog('############ Go into mteval-v11b calculation grouped by length ############')
    wlog('Calculating case-{}sensitive {}-gram BLEU ...'.format('' if cased else 'in', ngram))
    wlog('\tSource file: {}'.format(src_fpath))
    wlog('\tCandidate file: {}'.format(trans_fpath))
    wlog('\tReferences file:')
    for ref_fpath in ref_fpaths: wlog('\t\t{}'.format(ref_fpath))

    # split by source length
    src_F = open(src_fpath, 'r')
    src_lines = src_F.readlines()
    src_F.close()
    src_lines = [line.strip() for line in src_lines]

    hyp_F = open(trans_fpath, 'r')
    hyp_lines = hyp_F.readlines()
    hyp_F.close()
    hyp_lines = [line.strip() for line in hyp_lines]

    refs_lines = []
    for ref_fpath in ref_fpaths:
        ref_F = open(ref_fpath, 'r')
        ref_lines = ref_F.readlines()
        ref_F.close()
        ref_lines = [line.strip() for line in ref_lines]
        refs_lines.append(ref_lines)

    sent_num = len(src_lines)
    assert sent_num == len(hyp_lines) == len(refs_lines[0]), 'file length dismatch'

    ref_cnt = len(ref_fpaths)

    intervals = [(k, k + len_interval) for k in range(min_len, max_len, len_interval)]
    num_intervals = len(intervals)

    inter_hyps = [[] for i in range(num_intervals)]
    inter_refs = [ [[] for i in range(ref_cnt)] for k in range(num_intervals) ]

    for k in range(sent_num):

        src_line, hyp_line = src_lines[k], hyp_lines[k]
        src_L = len(src_line.split(' '))

        for interval_idx in range(num_intervals):
            left, right = intervals[interval_idx]
            if left < src_L and src_L <= right:
                inter_hyps[interval_idx].append(hyp_line)
                for ref_idx in range(ref_cnt):
                    ref_lines = refs_lines[ref_idx]
                    inter_refs[interval_idx][ref_idx].append(ref_lines[k])
            else:
                continue

    bleus = [None] * num_intervals

    for interval_idx in range(num_intervals):
        cand_lines = inter_hyps[interval_idx]
        refs_lines = [inter_refs[interval_idx][ref_idx] for ref_idx in range(ref_cnt)]
        result = mt_eval_bleu(cand_lines, refs_lines, ngram, cased=cased, char=char)
        result = float('%.2f' % (result * 100))
        bleus[interval_idx] = result

    return bleus


'''
convert some code of Moses mteval-v11b.pl into python code
'''
def token(s, cased=False):

    # language-independent part:
    s, n = re.subn('<skipped>', '', s)    # strip "skipped" tags
    s, n = re.subn('-\n', '', s)  # strip end-of-line hyphenation and join lines
    s, n = re.subn('\n', ' ', s)  # join lines
    s, n = re.subn('&quot;', '"', s)  # convert SGML tag for quote to "
    s, n = re.subn('&amp;', '&', s)   # convert SGML tag for ampersand to &
    s, n = re.subn('&lt;', '<', s)    # convert SGML tag for less-than to >
    s, n = re.subn('&gt;', '>', s)    # convert SGML tag for more-than to <

    # language-dependent part:
    s = ' ' + s + ' '
    # lowercase all characters (Case-insensitive BLEU)
    if cased is False: s = s.lower()

    # tokenize punctuation
    s, n = re.subn('([\{-\~\[-\` -\&\(-\+\:-\@\/])', lambda x: ' ' + x.group(0) + ' ', s)

    # tokenize period and comma unless preceded by a digit
    s, n = re.subn('([^0-9])([\.,])', lambda x: x.group(1) + ' ' + x.group(2) + ' ', s)

    # tokenize period and comma unless followed by a digit
    s, n = re.subn('([\.,])([^0-9])', lambda x: ' ' + x.group(1) + ' ' + x.group(2), s)

    # tokenize dash when preceded by a digit
    s, n = re.subn('([0-9])(-)', lambda x: x.group(1) + ' ' + x.group(2) + ' ', s)

    s, n = re.subn('\s+', ' ', s)    # only one space between words
    s, n = re.subn('^\s+', '', s)    # no leading space
    s, n = re.subn('\s+$', '', s)    # no trailing space

    return s

def merge_dict(d1, d2):
    '''
        Merge two dicts. The count of each item is the maximum count in two dicts.
    '''
    result = d1
    for key in d2:
        value = d2[key]
        if key in result:
            result[key] = max(result[key], value)
        else:
            result[key] = value
    return result

def sentence2dict(sentence, n):
    '''
        Count the number of n-grams in a sentence.

        :type sentence: string
        :param sentence: sentence text

        :type n: int
        :param n: maximum length of counted n-grams
    '''
    words = sentence.split(' ')
    result = {}
    for k in range(1, n + 1):
        for pos in range(len(words) - k + 1):
            gram = ' '.join(words[pos : pos + k])
            if gram in result:
                result[gram] += 1
            else:
                result[gram] = 1
    return result

def mt_eval_bleu(cand_lines, refs_lines, n=4, logfun=wlog, cased=False, char=False):
    '''
        Calculate BLEU score given translation and references.

        :type cand_lines: [sent_0, sent_1, ...]
        :param cand_lines: the translations

        :type refs_lines: [[ref0_sent_0, ref0_sent_1, ...], [ref1_sent_0, ref1_sent_1, ...], [], []]
        :param refs_lines: the list of references

        :type n: int
        :param n: maximum length of counted n-grams
    '''
    #cand_c="today weather very good", refs_lines=["today weather good", "would rain"],n=4
    correctgram_count = [0] * n
    ngram_count = [0] * n
    cand_length = 0
    ref_length = 0
    for sidx in range(len(cand_lines)):

        cand_line = cand_lines[sidx].strip(' ')
        if char is True: cand_line = ' '.join(zh_to_chars(cand_line.decode('utf-8')))
        else: cand_line = token(cand_line, cased)

        h_length = len(cand_line.split(' '))

        if char is True: refs = [' '.join(zh_to_chars(ref_lines[sidx].decode('utf-8'))) for ref_lines in refs_lines]
        else: refs = [token(ref_lines[sidx], cased) for ref_lines in refs_lines]

        # this is same with mteval-v11b.pl, using the length of the shortest reference
        ref_lengths = sorted([len(ref.split(' ')) for ref in refs])
        ref_length += ref_lengths[0]
        cand_length += h_length

        # why this ? more strict
        #cand_length += (h_length if h_length < ref_lengths[0] else ref_lengths[0])

        #print ref_lengths[0], ref_length, h_length, cand_length

        # another choice is use the minimal length difference of candthesis and four references !!
        #ref_distances = [abs(r - h_length) for r in ref_lengths]
        #ref_length += ref_lengths[numpy.argmin(ref_distances)]
        '''
        if sidx == 0:
            print h_length
            print ref_lengths[0]
            for ref_lines in refs_lines:
                print token(ref_lines[sidx]), len(token(ref_lines[sidx]).split(' '))
            print ref_lengths[numpy.argmin(ref_distances)]
        '''
        refs_dict = {}
        for ref in refs:  # four refs for one sentence
            ref_dict = sentence2dict(ref, n)
            refs_dict = merge_dict(refs_dict, ref_dict)

        #if sidx == 0:
        #    for key in refs_dict.keys():
        #        print key, refs_dict[key]
        cand_dict = sentence2dict(cand_line, n)

        for key in cand_dict:
            value = cand_dict[key]
            length = len(key.split(' '))
            ngram_count[length - 1] += value
            #if sidx == 0:
            #    print key, value, length
            #    print min(value, refs_dict[key])
            if key in refs_dict:
                correctgram_count[length - 1] += min(value, refs_dict[key])

    result = 0.
    bleu_n = [0.] * n
    #if correctgram_count[0] == 0: return 0.
    logfun('Total words count, ref {}, cand {}'.format(ref_length, cand_length))
    for i in range(n):
        logfun('{}-gram | ref {:8d} | match {:8d}'.format(i+1, ngram_count[i], correctgram_count[i]), 0)
        if correctgram_count[i] == 0:
            #correctgram_count[i] += 1
            #ngram_count[i] += 1
            logfun('')
            return 0.
        bleu_n[i] = correctgram_count[i] / ngram_count[i]
        logfun(' |\tPrecision: {}'.format(bleu_n[i]))
        result += math.log(bleu_n[i]) / n

    bp = 1.
    #bleu = geometric_mean(precisions) * bp     # same with mean function ?

    # there are no brevity penalty in mteval-v11b.pl, so with bp BLEU is a little lower
    if cand_length < ref_length:
        bp = math.exp(1 - ref_length / cand_length) if cand_length != 0 else 0

    BLEU = bp * math.exp(result)
    logfun('BP={}, ratio={}, BLEU={}'.format(bp, cand_length / ref_length, BLEU))

    return BLEU

def mteval_bleu_file(cand_file, ref_fpaths, ngram=4, cased=False, char=False):

    '''
        Calculate the BLEU score given translation files and reference files.

        :type cand_file: string
        :param ref_fpaths: the path to translation file
        :type ref_fpaths: list
        :param ref_fpaths: the list of path to reference files
    '''

    wlog('\n' + '#' * 30 + ' mteval-v11b ' + '#' * 30)
    wlog('Calculating case-{}sensitive {}-gram BLEU ...'.format('' if cased else 'in', ngram))
    wlog('\tcandidate file: {}'.format(cand_file))
    wlog('\treferences file:')
    for ref_fpath in ref_fpaths: wlog('\t\t{}'.format(ref_fpath))

    cand_f = open(cand_file, 'r')
    refs_f = [open(ref_fpath, 'r') for ref_fpath in ref_fpaths]

    cand_lines = cand_f.readlines()
    refs_lines = [ref_f.readlines() for ref_f in refs_f]
    result = mt_eval_bleu(cand_lines, refs_lines, ngram, cased=cased, char=char)
    result = float('%.2f' % (result * 100))

    cand_f.close()
    for ref_f in refs_f: ref_f.close()

    return result

if __name__ == "__main__":

    '''
    ref_fpaths = []
    for idx in range(4):
        #ref_fpath = '{}/{}'.format('work0', 'ref.seg.plain')
        #ref_fpath = '{}/{}'.format('data1', 'ref.seg.plain')
        #ref_fpath = '{}/{}'.format('data2', 'ref.seg.plain')
        #ref_fpath = '{}/{}'.format('data3', 'ref.seg.plain')
        ref_fpath = '{}{}'.format('/home/wen/3.corpus/segment_allnist_stanseg/nist03.ref', idx)
        if not os.path.exists(ref_fpath): continue
        ref_fpaths.append(ref_fpath)

    #print mteval_bleu_file('work0/hyp.seg.plain', ref_fpaths)
    #print mteval_bleu_file('data1/hyp.seg.plain', ref_fpaths)
    #print mteval_bleu_file('data2/hyp.seg.plain', ref_fpaths)
    #print mteval_bleu_file('data3/hyp.seg.plain', ref_fpaths)
    #print mteval_bleu_file('out', ref_fpaths)
    print mteval_bleu_file('trans_e10_upd15008_b10m1_bch1_32.64.txt', ref_fpaths)
    '''

    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='mt-eval BLEU score on multiple references.')
    parser.add_argument('-lc', help='Lowercase', action='store_true')
    parser.add_argument('-c', '--candidate', dest='c', required=True, help='translation file')
    parser.add_argument('-r', '--references', dest='r', required=True, help='reference_[0, 1, ...]')
    args = parser.parse_args()

    '''
    ref_fpaths = []
    ref_cnt = 4
    if ref_cnt == 1:
        ref_fpath = args.references
        if os.path.exists(ref_fpath): ref_fpaths.append(ref_fpath)
    else:
        for idx in range(ref_cnt):
            ref_fpath = '{}_{}'.format(args.references, idx)
            if not os.path.exists(ref_fpath): continue
            ref_fpaths.append(ref_fpath)
    '''

    # TODO: Multiple references
    #ref_fpaths = grab_all_trg_files('/home/wen/3.corpus/mt/mfd_1.25M/nist_test_new/mt06_u8.trg.tok.sb')
    ref_fpaths = grab_all_trg_files(args.r)

    #open_files = map(open, ref_fpaths)
    cand_file = args.c
    cased = ( not args.lc )
    mteval_bleu_file(cand_file, ref_fpaths, ngram=4, cased=cased)

    src_fpath = './nist03.src.plain.u8.a2b.stanseg'
    #src_fpath = './src.3'
    #bleus = length_bleu(src_fpath, ref_fpaths, cand_file, ngram=4, cased=cased)
    #print bleus








