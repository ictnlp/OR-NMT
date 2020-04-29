#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
import argparse
from math import exp, log
from functools import reduce
from collections import Counter

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
    #regex += [ur'^\s']

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

def ngram_count(words, n):
    if n <= len(words):
        return Counter(zip(*[words[i:] for i in range(n)]))
    return Counter()


def max_count(c1, c2):
    keys = set(list(c1) + list(c2))
    #keys = c1
    #for k in keys:
    #    print(k, max(c1[k], c2[k]))
    return Counter({k: max(c1[k], c2[k]) for k in keys})
    #return Counter({k: max(c1[k], c2[k]) for k in c1}) # problem here ...


def min_count(c1, c2):
    #keys = set(list(c1) + list(c2))
    #return Counter({k: max(c1[k], c2[k]) for k in keys})
    return Counter({k: min(c1[k], c2[k]) for k in c1})


def closest_min_length(candidate, references):
    l0 = len(candidate)
    return min((abs(len(r) - l0), len(r)) for r in references)[1]


def safe_log(n):
    if n <= 0:
        return -9999999999
        #return 0
    return log(n)

def precision_n(candidate, references, n):
    '''
    print(candidate)
    print(references)
    counts = []
    for ref in references:
        print('---------------------------------')
        print(ngram_count(ref, n)['sb'])
        counts.append(ngram_count(ref, n))
    ref_max = reduce(max_count, counts)
    '''
    ref_max = reduce(max_count, [ngram_count(ref, n) for ref in references])
    candidate_ngram_count = ngram_count(candidate, n)
    total = sum(candidate_ngram_count.values())
    correct = sum(reduce(min_count, (ref_max, candidate_ngram_count)).values())
    score = (correct / total) if total else 0
    return score, correct, total

def tokenize(txt, char=False):
    txt = txt.strip()
    if char is True: txt = zh_to_chars(txt.decode('utf-8'))
    else: txt = txt.split()
    return txt

def tokenize_lower(txt, char=False):
    txt = txt.strip().lower()
    if char is True: txt = zh_to_chars(txt.decode('utf-8'))
    else: txt = txt.split()
    return txt

def multi_bleu(cand_lines, refs_lines, tokenize_fn=tokenize, ngram=4, char=False):
    correct = [0] * ngram
    total = [0] * ngram
    cand_tot_length = 0
    ref_closest_length = 0
    assert len(cand_lines) == len(refs_lines[0]), \
            'dismatch len, cand {}, ref0_f {}'.format(len(cand_lines), len(refs_lines[0]))
    num, refs_num = len(cand_lines), len(refs_lines)

    #for candidate, references in zip(cand_lines, zip(*refs_lines)):
    for k in range(num):
        candidate = cand_lines[k]
        candidate = tokenize_fn(candidate, char)
        references = []
        for ref_idx in range(refs_num):
            references.append(tokenize_fn(refs_lines[ref_idx][k], char))
        #references = map(tokenize_fn, references, [char for _ in references])
        #print(candidate)
        #print(references)
        cand_tot_length += len(candidate)
        ref_closest_length += closest_min_length(candidate, references)
        for n in range(ngram):
            sc, cor, tot = precision_n(candidate, references, n + 1)
            correct[n] += cor
            total[n] += tot

    precisions = [(correct[n] / total[n]) if correct[n] else 0 for n in range(ngram)]

    if cand_tot_length < ref_closest_length:
        brevity_penalty = exp(1 - ref_closest_length / cand_tot_length) if cand_tot_length != 0 else 0
    else:
        brevity_penalty = 1
    #print(precisions[1])
    score = 100 * brevity_penalty * exp(
        sum(safe_log(precisions[n]) for n in range(ngram)) / ngram)
    prec_pc = [100 * p for p in precisions]
    '''
    for n in range(ngram):
        if total[n] == 0:   # back to n-gram BLEU
            if n == 0:
                wlog('What ? Null references ... ')
                bleu, prec_pc = 0., 0.
                break
            score = 100 * brevity_penalty * exp(
                sum(safe_log(precisions[t]) for t in range(n)) / n)
            prec_pc = [100 * precisions[t] for t in range(n)]
            break
    '''

    return score, prec_pc, brevity_penalty, cand_tot_length, ref_closest_length

def multi_bleu_file(cand_file, ref_fpaths, cased=False, ngram=4, char=False):

    wlog('\n' + '#' * 30 + ' multi-bleu ' + '#' * 30)
    tokenize_fn = tokenize if cased is True else tokenize_lower
    wlog('Calculating case-{}sensitive tokenized {}-gram BLEU ...'.format('' if cased else 'in', ngram))
    wlog('\tcandidate file: {}'.format(cand_file))
    wlog('\treferences file:')
    for ref in ref_fpaths: wlog('\t\t{}'.format(ref))

    cand_f = open(cand_file, 'r')
    refs_f = [open(ref_fpath, 'r') for ref_fpath in ref_fpaths]

    cand_lines = cand_f.readlines()
    refs_lines = [ref_f.readlines() for ref_f in refs_f]
    score, precisions, brevity_penalty, cand_tot_length, ref_closest_length = \
        multi_bleu(cand_lines, refs_lines, tokenize_fn, ngram, char=char)

    cand_f.close()
    for ref_f in refs_f: ref_f.close()

    precs_list = []
    for prec in precisions: precs_list.append('{:.1f}'.format(prec))
    wlog('BLEU = {:.2f}, {} (BP={:.3f}, ratio={:.3f}, hyp_len={:d}, ref_len={:d})\n'.format(
            score, '/'.join(precs_list), brevity_penalty,
        cand_tot_length / ref_closest_length, cand_tot_length, ref_closest_length))

    score = float('%.2f' % (score))
    return score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BLEU score on multiple references.')
    parser.add_argument('-lc', help='Lowercase', action='store_true')
    parser.add_argument('-c', '--candidate', dest='c', required=True, help='translation file')
    parser.add_argument('-r', '--references', dest='r', required=True, help='reference[0, 1, ...]')
    args = parser.parse_args()

    '''
    ref_fpaths = []
    ref_cnt = 2
    if ref_cnt == 1:
        ref_fpath = args.reference
        if os.path.exists(ref_fpath): ref_fpaths.append(ref_fpath)
    else:
        for idx in range(ref_cnt):
            ref_fpath = '{}_{}'.format(args.reference, idx)
            if not os.path.exists(ref_fpath): continue
            ref_fpaths.append(ref_fpath)
    '''
    # TODO: Multiple references
    #ref_fpaths = grab_all_trg_files('/home/wen/3.corpus/mt/mfd_1.25M/nist_test_new/mt06_u8.trg.tok.sb')
    ref_fpaths = grab_all_trg_files(args.r)

    #open_files = map(open, ref_fpaths)
    cand_file = args.c
    cased = ( not args.lc )
    multi_bleu_file(cand_file, ref_fpaths, cased, 4)



