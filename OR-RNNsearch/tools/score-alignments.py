#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division

import optparse
import sys
import uniout

optparser = optparse.OptionParser()
optparser.add_option("-d", "--prefix", dest="prefix", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-s", "--src", dest="source", default="src", help="Suffix of source filename (default=src)")
optparser.add_option("-t", "--trg", dest="target", default="trg", help="Suffix of target filename (default=trg)")
optparser.add_option("-g", "--gold", dest="gold", default="a", help="Suffix of gold alignments filename (default=a)")
optparser.add_option("-i", "--hypo", dest="hypo", default="ha", help="hypo alignments filename (default=ha)")
optparser.add_option("-n", "--num_display", dest="n", default=sys.maxint, type="int", help="Number of alignments to display")
(opts, args) = optparser.parse_args()

src_data = "%s.%s" % (opts.prefix, opts.source)
trg_data = "%s.%s" % (opts.prefix, opts.target)
gold_aln = "%s.%s" % (opts.prefix, opts.gold)
hypo_aln = opts.hypo


def getAln(aln_str):

    map_aln = {}
    for aln_item in aln_str.strip().split():
        if aln_item.find(':') < 0: continue
        if aln_item.find('/') < 0: map_aln[aln_item] = 1
        else:
            aln_type = aln_item.split('/')
            map_aln[aln_type[0]] = aln_type[1]

    return map_aln


ref_alns, tst_alns = [], []
total_sure, total_actual, total_match_sure, total_match_possible = 0, 0, 0, 0

for (sent_id, (s, t, g, h)) in enumerate(zip(open(src_data), open(trg_data), open(gold_aln), open(hypo_aln))):

    _sure = set([tuple(map(int, x.split("/")[0].split(':') if x.rfind('/') > 0 \
                           else x.split(':') )) for x in filter(lambda x: x.find(":") > -1,
                                                                g.strip().split())])

    _possible = set([tuple(map(int, x.split("/")[0].split('?') if x.rfind('/') > 0 \
                               else x.split('?') )) for x in filter(lambda x: x.find("?") > -1,
                                                                    g.strip().split())])

    _alignment = set([tuple(map(int, x.split("/")[0].split(':') if x.rfind('/') > 0 \
                                else x.split(':') )) for x in filter(lambda x: x.find(":") > -1,
                                                                     h.strip().split())])
    src_words = s.strip().split()    # source
    trg_words = t.strip().split()    # target

    gold_aln_map = getAln(g)
    hypo_aln_map = getAln(h)

    # calculate precision, recall and AER for a sentence
    sure, actual, match_sure, match_possible = 0, 0, 0, 0

    sure = len(filter(lambda x: x == '1', gold_aln_map.values()))
    actual = len(hypo_aln_map)
    for aln_item, _ in hypo_aln_map.items():
        if aln_item in gold_aln_map:
            match_possible += 1
            if gold_aln_map[aln_item] == '1':
                match_sure += 1

    precision = match_possible / actual
    recall = match_sure / sure
    AER = 1.0 - (match_sure + match_possible) / (actual + sure)

    sys.stderr.write('[{}], precision: {}={}/{}, recall: {}={}/{}, AER: {}\n'.format(
        sent_id+1, precision, match_possible, actual, recall, match_sure, sure, AER))

    total_sure += sure
    total_actual += actual
    total_match_sure += match_sure
    total_match_possible += match_possible

    if (sent_id < opts.n):
        sys.stdout.write('Sent {}, KEY: ( ) = guessed, * = sure, ? = possible\n'.format(sent_id+1))
        sys.stdout.write("  ")
        for j in trg_words:    # target for x-axis
          sys.stdout.write("---")
        sys.stdout.write("\n")
        for (i, e_i) in enumerate(src_words):  # source
          sys.stdout.write(" |")
          for (j, _) in enumerate(trg_words):  # target
            (left, right) = ("(", ")") if (i+1, j+1) in _alignment else (" ", " ")
            point = "*" if (i+1, j+1) in _sure else "?" if (i+1, j+1) in _possible else " "
            sys.stdout.write("%s%s%s" % (left,point,right))
          sys.stdout.write(" | %s\n" % e_i)
        sys.stdout.write("  ")
        for j in trg_words:    # target for x-axis
          sys.stdout.write("---")
        sys.stdout.write("\n")
        #maxL = max([len(eword.decode('utf-8')) for eword in src_words])
        #for k in range(maxL):
        for k in range(max(map(len, trg_words))):
          sys.stdout.write("  ")
          for word in trg_words:
            letter = word[k] if len(word) > k else " "
            sys.stdout.write(" %s " % letter)
            #char = word.decode('utf-8')[k] if len(word.decode('utf-8')) > k else "  "
            #sys.stdout.write(" %s " % char)
          sys.stdout.write("\n")
        sys.stdout.write("\n")

precision = total_match_possible / total_actual
recall = total_match_sure / total_sure
AER = 1.0 - (total_match_sure + total_match_possible) / (total_actual + total_sure)

sys.stderr.write('Precision = {}={}/{}\nRecall = {}={}/{}\nAER = {}'.format(
    precision, total_match_possible, total_actual, recall, total_match_sure, total_sure, AER))

#'{:.2f}'.format(bb * 100)

#for _ in (sys.stdin): # avoid pipe error
#  pass






