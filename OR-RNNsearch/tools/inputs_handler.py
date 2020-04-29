# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import

import io
import os
import sys
import math
import numpy
import torch as tc
from collections import defaultdict

import wargs
from tools.utils import *
from tools.inputs import *
from tools.vocab import Vocab
from tools.mteval_bleu import zh_to_chars

def extract_vocab(data_file, vocab_file, max_vcb_size=30000, max_seq_len=50, char=False):

    wlog('\tmax length {}, char? {}'.format(max_seq_len, char))
    if os.path.exists(vocab_file) is True:

        # If vocab file has been exist, we load word vocabulary
        wlog('Load vocabulary from file {}'.format(vocab_file))
        vocab = Vocab()
        vocab.load_from_file(vocab_file)

    else:

        vocab = count_vocab(data_file, max_vcb_size, max_seq_len, char=char)
        vocab.write_into_file(vocab_file)
        wlog('Save vocabulary file into {}'.format(vocab_file))

    return vocab

def count_vocab(data_file, max_vcb_size, max_seq_len=50, char=False):

    assert data_file and os.path.exists(data_file), 'need file to extract vocabulary ...'

    vocab = Vocab()
    #with open(data_file, 'r') as f:
    with io.open(data_file, encoding='utf-8') as f:
        for sent in f.readlines():
            #sent = sent.strip().encode('utf-8')
            sent = sent.strip()
            if char is True: words = zh_to_chars(sent)
            else: words = sent.split()
            if len(words) > max_seq_len: continue
            for word in words: vocab.add(word)

    words_cnt = sum(vocab.freq.values())
    new_vocab, new_words_cnt = vocab.keep_vocab_size(max_vcb_size)
    wlog('|Final vocabulary| / |Original vocabulary| = {} / {} = {:4.2f}%'
         .format(new_words_cnt, words_cnt, (new_words_cnt/words_cnt) * 100))

    new_vocab.idx2key = {k: str(v) for k, v in new_vocab.idx2key.items()}
    new_vocab.key2idx = {str(k): v for k, v in new_vocab.key2idx.items()}

    return new_vocab

def wrap_data(data_dir, file_prefix, src_suffix, trg_prefix, src_vocab, trg_vocab,
              shuffle=True, sort_k_batches=1, max_seq_len=50, char=False):

    srcF = open(os.path.join(data_dir, '{}.{}'.format(file_prefix, src_suffix)), 'r')
    num = len(srcF.readlines())
    srcF.close()
    point_every, number_every = int(math.ceil(num/100)), int(math.ceil(num/10))

    srcF = io.open(os.path.join(data_dir, '{}.{}'.format(file_prefix, src_suffix)),
                   mode='r', encoding='utf-8')

    trgFs = []  # maybe have multi-references for valid, we open them together
    for fname in os.listdir(data_dir):
        if fname.startswith('{}.{}'.format(file_prefix, trg_prefix)):
            wlog('\t{}'.format(os.path.join(data_dir, fname)))
            trgFs.append(open(os.path.join(data_dir, fname), 'r'))
    wlog('NOTE: Target side has {} references.'.format(len(trgFs)))

    idx, ignore, longer = 0, 0, 0
    srcs, trgs, slens = [], [], []
    while True:

        src_sent = srcF.readline().strip()
        if not src_sent:
            wlog('\nFinish to read bilingual corpus.')
            break

        if char is True: src_sent = ' '.join(zh_to_chars(src_sent))
        trg_refs = [trgF.readline().strip() for trgF in trgFs]

        if src_sent == '' and all([trg_ref == '' for trg_ref in trg_refs]):
            continue

        idx += 1
        if ( idx % point_every ) == 0:
            wlog('.', newline=0)
            sys.stderr.flush()
        if ( idx % number_every ) == 0: wlog(idx, newline=0)

        if src_sent == '' or any([trg_ref == '' for trg_ref in trg_refs]):
            wlog('Ignore abnormal blank sentence in line number {}'.format(idx))
            ignore += 1
            continue

        src_words = src_sent.split()
        src_len = len(src_words)
        trg_refs_words = [trg_ref.split() for trg_ref in trg_refs]
        if src_len <= max_seq_len and all([len(tws) <= max_seq_len for tws in trg_refs_words]):

            src_tensor = [ src_vocab.keys2idx(src_words, UNK_WORD) ]
            trg_refs_tensor = [trg_vocab.keys2idx(trg_ref_words, UNK_WORD,
                                              bos_word=BOS_WORD, eos_word=EOS_WORD)
                               for trg_ref_words in trg_refs_words]

            srcs.append(src_tensor)
            trgs.append(trg_refs_tensor)
            slens.append(src_len)
        else:
            longer += 1

    srcF.close()
    for trgF in trgFs: trgF.close()

    train_size = len(srcs)
    assert train_size == idx - ignore - longer, 'Wrong .. '
    wlog('Sentence-pairs count: {}(total) - {}(ignore) - {}(longer) = {}'.format(
        idx, ignore, longer, idx - ignore - longer))

    if shuffle is True:

        #assert len(trgFs) == 1, 'Unsupport to shuffle validation set.'
        wlog('Shuffling the whole dataset ... ', False)
        rand_idxs = tc.randperm(train_size).tolist()
        srcs = [srcs[k] for k in rand_idxs]
        trgs = [trgs[k] for k in rand_idxs]
        slens = [slens[k] for k in rand_idxs]
        wlog('done.')

    final_srcs, final_trgs = sort_batches(srcs, trgs, slens, wargs.batch_size, sort_k_batches)

    return final_srcs, final_trgs

def wrap_tst_data(src_data, src_vocab, char=False):

    srcs, slens = [], []
    srcF = io.open(src_data, mode='r', encoding='utf-8')
    idx = 0

    while True:

        src_sent = srcF.readline()
        if src_sent == '':
            wlog('\nFinish to read monolingual test dataset {}, count {}'.format(src_data, idx))
            break
        idx += 1

        if src_sent == '':
            wlog('Error. Ignore abnormal blank sentence in line number {}'.format(idx))
            sys.exit(0)

        src_sent = src_sent.strip()
        if char is True: src_sent = ' '.join(zh_to_chars(src_sent))
        src_words = src_sent.split()
        src_len = len(src_words)

        src_tensor = [ src_vocab.keys2idx(src_words, UNK_WORD) ]

        srcs.append(src_tensor)
        slens.append(src_len)

    srcF.close()
    return srcs, slens


if __name__ == "__main__":

    src = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_src_suffix))
    trg = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_trg_suffix))
    vocabs = {}
    wlog('\nPreparing source vocabulary from {} ... '.format(src))
    src_vocab = extract_vocab(src, wargs.src_vcb, wargs.n_src_vcb_plan, wargs.max_seq_len)
    wlog('\nPreparing target vocabulary from {} ... '.format(trg))
    trg_vocab = extract_vocab(trg, wargs.trg_vcb, wargs.n_trg_vcb_plan, wargs.max_seq_len)
    n_src_vcb, n_trg_vcb = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(n_src_vcb, n_trg_vcb))
    vocabs['src'], vocabs['trg'] = src_vocab, trg_vocab

    wlog('\nPreparing training set from {} and {} ... '.format(src, trg))
    trains = {}
    train_src_tlst, train_trg_tlst = wrap_data(wargs.dir_data, wargs.train_prefix,
                                               wargs.train_src_suffix, wargs.train_trg_suffix,
                                               src_vocab, trg_vocab, max_seq_len=wargs.max_seq_len)
    assert len(train_trg_tlst[0]) == 1, 'Require only one reference in training dataset.'
    '''
    list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...]
    no padding
    '''

    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)
    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))

    batch_valid = None
    if wargs.val_prefix is not None:
        val_src_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_src_suffix)
        val_trg_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_ref_suffix)
        wlog('\nPreparing validation set from {} and {} ... '.format(val_src_file, val_trg_file))
        valid_src_tlst, valid_trg_tlst = wrap_data(wargs.val_tst_dir, wargs.val_prefix,
                                                   wargs.val_src_suffix, wargs.val_ref_suffix,
                                                   src_vocab, trg_vocab, shuffle=False,
                                                   max_seq_len=wargs.dev_max_seq_len)
        batch_valid = Input(valid_src_tlst, valid_trg_tlst, 1, volatile=True, batch_sort=False)

    batch_tests = None
    if wargs.tests_prefix is not None:
        assert isinstance(wargs.tests_prefix, list), 'Test files should be list.'
        init_dir(wargs.dir_tests)
        batch_tests = {}
        for prefix in wargs.tests_prefix:
            init_dir(wargs.dir_tests + '/' + prefix)
            test_file = '{}{}.{}'.format(wargs.val_tst_dir, prefix, wargs.val_src_suffix)
            wlog('\nPreparing test set from {} ... '.format(test_file))
            test_src_tlst, _ = wrap_tst_data(test_file, src_vocab)
            batch_tests[prefix] = Input(test_src_tlst, None, 1, volatile=True)

    inputs = {}
    inputs['vocab'] = vocabs
    inputs['train'] = batch_train
    inputs['valid'] = batch_valid
    inputs['tests'] = batch_tests

    wlog('Saving data to {} ... '.format(wargs.inputs_data), False)
    tc.save(inputs, wargs.inputs_data)
    wlog('\n## Finish to Prepare Dataset ! ##\n')












