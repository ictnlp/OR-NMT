from __future__ import division

import sys
import math
import numpy
import itertools

from bleu import bleu

#sys.path.append('../')
from utils import debug, wlog

# usage: signtest_bleu.py -b <hypo_base> -m <hypo_model> -r <ref_0 ref_1 ...>
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='signtest BLEU score on two systems.')
    parser.add_argument('-b', '--baseline', dest='b', required=True, help='translation from baseline model')
    parser.add_argument('-m', '--model', dest='m', required=True, help='translation from other model')
    parser.add_argument('-r', '--references', dest='r', required=True, nargs='+', help='Reads the reference_[0, 1, ...]')
    parser.add_argument('-lc', help='Lowercase', action='store_true')
    parser.add_argument('-v', help='print log', action='store_true')

    args = parser.parse_args()

    hypo_b = open(args.b, 'r').read().strip()
    hypo_m = open(args.m, 'r').read().strip()
    refs = [open(ref_fpath, 'r').read().strip() for ref_fpath in args.r]

    cased = ( not args.lc )
    bleu_b = bleu(hypo_b, refs, 4, cased=cased)
    bleu_m = bleu(hypo_m, refs, 4, cased=cased)
    wlog('Baseline BLEU: {:4.2f}'.format(bleu_b))
    wlog('Model BLEU   : {:4.2f}'.format(bleu_m))

    list_hypo_b, list_hypo_m = hypo_b.split('\n'), hypo_m.split('\n')
    better = worse = 0
    fake = list_hypo_b[:]
    assert len(list_hypo_b) == len(list_hypo_m), 'Length mismatch ... '

    num = len(list_hypo_b)
    point_every, number_every = int(math.ceil(num/100)), int(math.ceil(num/10))

    for i in xrange(len(fake)):

        fake[i] = list_hypo_m[i]
        fake_score = bleu('\n'.join(fake), refs, 4, logfun=debug, cased=cased)

        if fake_score > bleu_b: better += 1
        elif fake_score < bleu_b: worse += 1

        if args.v is True:
            wlog('sentence {} {} {}'.format(i, bleu_b, fake_score))

        fake[i] = list_hypo_b[i]
        if numpy.mod(i + 1, point_every) == 0: wlog('.', False)
        if numpy.mod(i + 1, number_every) == 0: wlog('{}'.format(idx + 1), False)

    wlog('Model better on {} sentences, worse on {} sentences'.format(better, worse))

    n = better + worse
    mean = better / n
    se = math.sqrt( mean * (1. - mean) / n )

    left_95, right_95 = mean-1.96*se, mean+1.96*se
    left_99, right_99 = mean-2.58*se, mean+2.58*se

    wlog('Pr(better|different) = {}'.format(mean))
    wlog('95% confidence interval: ({}, {})'.format(left_95, right_95))
    wlog('99% confidence interval: ({}, {})'.format(left_99, right_99))

    if left_99 > 0.5:
         wlog('Model is significantly better (p < 0.01)')
    elif right_99 < 0.5:
         wlog('Model is significantly worse (p < 0.01)')
    elif left_95 > 0.5:
         wlog('Model is significantly better (p < 0.05)')
    elif right_95 < 0.5:
         wlog('Model is significantly worse (p < 0.05)')
    else:
         wlog('No significant difference')



