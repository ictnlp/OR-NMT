from __future__ import division

import numpy as np
import torch as tc
from torch import cuda

import wargs
from inputs_handler import *
from tools.inputs import Input
from tools.optimizer import Optim
from tools.utils import init_dir, wlog, _load_model

if wargs.model == 0: from models.groundhog import *
elif wargs.model == 1: from models.rnnsearch import *
elif wargs.model == 2: from models.rnnsearch_ia import *
elif wargs.model == 3: from models.ran_agru import *
elif wargs.model == 4: from models.rnnsearch_rn import *
elif wargs.model == 5: from models.nmt_sru import *
elif wargs.model == 6: from models.nmt_cyk import *
elif wargs.model == 7: from models.non_local import *

from translate import Translator
from car_trainer import Trainer

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

class DataHisto():

    def __init__(self, chunk_D0):

        self.chunk_Ds = chunk_D0
        assert len(chunk_D0[0]) == len(chunk_D0[1])
        self.size = len(chunk_D0[0])

    def add_batch_data(self, chunk_Dk):

        self.chunk_Ds = (self.chunk_Ds[0] + chunk_Dk[0],
                         self.chunk_Ds[1] + chunk_Dk[1])
        self.size += len(chunk_Dk[0])

    def merge_batch(self, new_batch, batch_sort=False):

        # fine selectly sampling from history training chunks
        sample_xs, sample_ys = [], []

        '''
        while not len(sample_xs) == wargs.batch_size:
            k = np.random.randint(0, self.size, (1,))[0]
            srcL, trgL = len(self.chunk_Ds[0][k]), len(self.chunk_Ds[1][k])
            neg = (srcL<10) or (srcL>wargs.max_seq_len) or (trgL<10) or (trgL>wargs.max_seq_len)
            if neg: continue
            sample_xs.append(self.chunk_Ds[0][k])
            sample_ys.append(self.chunk_Ds[1][k])
        '''
        ids = np.random.randint(0, self.size, (wargs.batch_size,))
        for idx in ids: # firstly we randomly select a batch data from the history large dataset
            sample_xs.append(self.chunk_Ds[0][idx])
            sample_ys.append(self.chunk_Ds[1][idx])

        batch_src, batch_trg = [], []
        #shuf_idx = tc.randperm(new_batch[1].size(1))
        #for idx in range(new_batch[1].size(1) / 2):
        for idx in range(new_batch[1].size(1)):  # add another new batch data
            #src = tc.Tensor(sent_filter(new_batch[1][:, idx].data.tolist()))
            #trg = tc.Tensor(sent_filter(new_batch[2][0][:, idx].data.tolist()))
            src = tc.Tensor(new_batch[1][:, idx].data.tolist())
            trg = tc.Tensor(new_batch[2][0][:, idx].data.tolist())
            sample_xs.append(src)
            sample_ys.append([trg])

        return Input(sample_xs, sample_ys, wargs.batch_size * 2, batch_sort=batch_sort, printlog=False)

def main():

    # Check if CUDA is available
    if cuda.is_available():
        wlog('CUDA is available, specify device by gpu_id argument (i.e. gpu_id=[3])')
    else:
        wlog('Warning: CUDA is not available, try CPU')

    if wargs.gpu_id:
        cuda.set_device(wargs.gpu_id[0])
        wlog('Using GPU {}'.format(wargs.gpu_id[0]))

    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)

    '''
    train_srcD_file = wargs.dir_data + 'train.10k.zh5'
    wlog('\nPreparing source vocabulary from {} ... '.format(train_srcD_file))
    src_vocab = extract_vocab(train_srcD_file, wargs.src_dict, wargs.src_dict_size)

    train_trgD_file = wargs.dir_data + 'train.10k.en5'
    wlog('\nPreparing target vocabulary from {} ... '.format(train_trgD_file))
    trg_vocab = extract_vocab(train_trgD_file, wargs.trg_dict, wargs.trg_dict_size)

    train_src_file = wargs.dir_data + 'train.10k.zh0'
    train_trg_file = wargs.dir_data + 'train.10k.en0'
    wlog('\nPreparing training set from {} and {} ... '.format(train_src_file, train_trg_file))
    train_src_tlst, train_trg_tlst = wrap_data(train_src_file, train_trg_file, src_vocab, trg_vocab)
    #list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...], no padding
    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))
    src_vocab_size, trg_vocab_size = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))
    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)
    '''

    src = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_src_suffix))
    trg = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_trg_suffix))
    vocabs = {}
    wlog('\nPreparing source vocabulary from {} ... '.format(src))
    src_vocab = extract_vocab(src, wargs.src_dict, wargs.src_dict_size)
    wlog('\nPreparing target vocabulary from {} ... '.format(trg))
    trg_vocab = extract_vocab(trg, wargs.trg_dict, wargs.trg_dict_size)
    src_vocab_size, trg_vocab_size = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))
    vocabs['src'], vocabs['trg'] = src_vocab, trg_vocab

    wlog('\nPreparing training set from {} and {} ... '.format(src, trg))
    trains = {}
    train_src_tlst, train_trg_tlst = wrap_data(wargs.dir_data, wargs.train_prefix,
                                               wargs.train_src_suffix, wargs.train_trg_suffix,
                                               src_vocab, trg_vocab, max_seq_len=wargs.max_seq_len)
    '''
    list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...]
    no padding
    '''
    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size, batch_sort=True)
    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))

    batch_valid = None
    if wargs.val_prefix is not None:
        val_src_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_src_suffix)
        val_trg_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_ref_suffix)
        wlog('\nPreparing validation set from {} and {} ... '.format(val_src_file, val_trg_file))
        valid_src_tlst, valid_trg_tlst = wrap_data(wargs.val_tst_dir, wargs.val_prefix,
                                                   wargs.val_src_suffix, wargs.val_ref_suffix,
                                                   src_vocab, trg_vocab,
                                                   shuffle=False, sort_data=False,
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
            batch_tests[prefix] = Input(test_src_tlst, None, 1, volatile=True, batch_sort=False)
    wlog('\n## Finish to Prepare Dataset ! ##\n')

    nmtModel = NMT(src_vocab_size, trg_vocab_size)
    if wargs.pre_train is not None:

        assert os.path.exists(wargs.pre_train)

        _dict = _load_model(wargs.pre_train)
        # initializing parameters of interactive attention model
        class_dict = None
        if len(_dict) == 4: model_dict, eid, bid, optim = _dict
        elif len(_dict) == 5:
            model_dict, class_dict, eid, bid, optim = _dict
        for name, param in nmtModel.named_parameters():
            if name in model_dict:
                param.requires_grad = not wargs.fix_pre_params
                param.data.copy_(model_dict[name])
                wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.weight'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.weight'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.bias'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.bias'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            else: init_params(param, name, True)

        wargs.start_epoch = eid + 1

    else:
        for n, p in nmtModel.named_parameters(): init_params(p, n, True)
        optim = Optim(
            wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )
        optim.init_optimizer(nmtModel.parameters())

    if wargs.gpu_id:
        wlog('Push model onto GPU {} ... '.format(wargs.gpu_id), 0)
        nmtModel.cuda()
    else:
        wlog('Push model onto CPU ... ', 0)
        nmtModel.cpu()

    wlog('done.')
    wlog(nmtModel)
    wlog(optim)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('Parameters number: {}/{}'.format(pcnt1, pcnt2))

    trainer = Trainer(nmtModel, src_vocab.idx2key, trg_vocab.idx2key, optim, trg_vocab_size,
                      valid_data=batch_valid, tests_data=batch_tests)

    # add 1000 to train
    train_all_chunks = (train_src_tlst, train_trg_tlst)
    dh = DataHisto(train_all_chunks)

    '''
    dev_src0 = wargs.dir_data + 'dev.1k.zh0'
    dev_trg0 = wargs.dir_data + 'dev.1k.en0'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src0, dev_trg0))
    dev_src0, dev_trg0 = wrap_data(dev_src0, dev_trg0, src_vocab, trg_vocab)
    wlog(len(train_src_tlst))

    dev_src1 = wargs.dir_data + 'dev.1k.zh1'
    dev_trg1 = wargs.dir_data + 'dev.1k.en1'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src1, dev_trg1))
    dev_src1, dev_trg1 = wrap_data(dev_src1, dev_trg1, src_vocab, trg_vocab)

    dev_src2 = wargs.dir_data + 'dev.1k.zh2'
    dev_trg2 = wargs.dir_data + 'dev.1k.en2'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src2, dev_trg2))
    dev_src2, dev_trg2 = wrap_data(dev_src2, dev_trg2, src_vocab, trg_vocab)

    dev_src3 = wargs.dir_data + 'dev.1k.zh3'
    dev_trg3 = wargs.dir_data + 'dev.1k.en3'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src3, dev_trg3))
    dev_src3, dev_trg3 = wrap_data(dev_src3, dev_trg3, src_vocab, trg_vocab)

    dev_src4 = wargs.dir_data + 'dev.1k.zh4'
    dev_trg4 = wargs.dir_data + 'dev.1k.en4'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src4, dev_trg4))
    dev_src4, dev_trg4 = wrap_data(dev_src4, dev_trg4, src_vocab, trg_vocab)
    wlog(len(dev_src4+dev_src3+dev_src2+dev_src1+dev_src0))
    batch_dev = Input(dev_src4+dev_src3+dev_src2+dev_src1+dev_src0, dev_trg4+dev_trg3+dev_trg2+dev_trg1+dev_trg0, wargs.batch_size)
    '''

    batch_dev = None
    assert wargs.dev_prefix is not None, 'Requires development to tuning.'
    dev_src_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.dev_prefix, wargs.val_src_suffix)
    dev_trg_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.dev_prefix, wargs.val_ref_suffix)
    wlog('\nPreparing dev set from {} and {} ... '.format(dev_src_file, dev_trg_file))
    valid_src_tlst, valid_trg_tlst = wrap_data(wargs.val_tst_dir, wargs.dev_prefix,
                                               wargs.val_src_suffix, wargs.val_ref_suffix,
                                               src_vocab, trg_vocab,
                                               shuffle=True, sort_data=True,
                                               max_seq_len=wargs.dev_max_seq_len)
    batch_dev = Input(valid_src_tlst, valid_trg_tlst, wargs.batch_size, batch_sort=True)

    trainer.train(dh, batch_dev, 0, merge=True, name='DH_{}'.format('dev'))

    '''
    chunk_size = 1000
    rand_ids = tc.randperm(len(train_src_tlst))[:chunk_size * 1000]
    rand_ids = rand_ids.split(chunk_size)
    #train_chunks = [(dev_src, dev_trg)]
    train_chunks = []
    for k in range(len(rand_ids)):
        rand_id = rand_ids[k]
        chunk_src_tlst = [train_src_tlst[i] for i in rand_id]
        chunk_trg_tlst = [train_trg_tlst[i] for i in rand_id]
        #wlog('Sentence-pairs count in training data: {}'.format(len(src_samples_train)))
        #batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)
        #batch_train = Input(src_samples_train, trg_samples_train, wargs.batch_size)
        train_chunks.append((chunk_src_tlst, chunk_trg_tlst))

    chunk_D0 = train_chunks[0]
    dh = DataHisto(chunk_D0)
    c0_input = Input(chunk_D0[0], chunk_D0[1], wargs.batch_size)
    trainer.train(dh, c0_input, 0, batch_valid, batch_tests, merge=False, name='DH_{}'.format(0))
    for k in range(1, len(train_chunks)):
        wlog('*' * 30, False)
        wlog(' Next Data {} '.format(k), False)
        wlog('*' * 30)
        chunk_Dk = train_chunks[k]
        ck_input = Input(chunk_Dk[0], chunk_Dk[1], wargs.batch_size)
        trainer.train(dh, ck_input, k, batch_valid, batch_tests, merge=True, name='DH_{}'.format(k))
        dh.add_batch_data(chunk_Dk)
    '''



if __name__ == "__main__":

    main()















