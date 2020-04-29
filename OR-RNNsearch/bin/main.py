#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())

import torch as tc
from torch import cuda

import wargs
from tools.inputs_handler import *
from tools.inputs import Input
from tools.optimizer import Optim
from models.losser import Classifier
from models.embedding import WordEmbedding
from models.model_builder import build_NMT
from tools.utils import init_dir, wlog

# Check if CUDA is available
if cuda.is_available():
    wlog('CUDA is available, specify device by gpu_id argument (i.e. gpu_id=[0, 1, 2])')
else:
    wlog('Warning: CUDA is not available, train on CPU')

if wargs.gpu_id is not None:
    #cuda.set_device(wargs.gpu_id[0])
    device = tc.device('cuda:{}'.format(wargs.gpu_id[0]) if cuda.is_available() else 'cpu')
    wlog('Set device {}, will use {} GPUs {}'.format(
        wargs.gpu_id[0], len(wargs.gpu_id), wargs.gpu_id))

from trainer import *

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

tc.manual_seed(1111)

def main():

    #if wargs.ss_type is not None: assert wargs.model == 1, 'Only rnnsearch support schedule sample'
    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)

    src = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_src_suffix))
    trg = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_trg_suffix))
    vocabs = {}
    wlog('\nPreparing source vocabulary from {} ... '.format(src))
    src_vocab = extract_vocab(src, wargs.src_vcb, wargs.n_src_vcb_plan,
                              wargs.max_seq_len, char=wargs.src_char)
    wlog('\nPreparing target vocabulary from {} ... '.format(trg))
    trg_vocab = extract_vocab(trg, wargs.trg_vcb, wargs.n_trg_vcb_plan, wargs.max_seq_len)
    n_src_vcb, n_trg_vcb = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(n_src_vcb, n_trg_vcb))
    vocabs['src'], vocabs['trg'] = src_vocab, trg_vocab

    wlog('\nPreparing training set from {} and {} ... '.format(src, trg))
    trains = {}
    train_src_tlst, train_trg_tlst = wrap_data(wargs.dir_data, wargs.train_prefix,
                                               wargs.train_src_suffix, wargs.train_trg_suffix,
                                               src_vocab, trg_vocab, shuffle=True,
                                               sort_k_batches=wargs.sort_k_batches,
                                               max_seq_len=wargs.max_seq_len,
                                               char=wargs.src_char)
    '''
    list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...]
    no padding
    '''
    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size,
                        batch_type=wargs.batch_type, bow=wargs.trg_bow, batch_sort=False)
    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))

    batch_valid = None
    if wargs.val_prefix is not None:
        val_src_file = os.path.join(wargs.val_tst_dir, '{}.{}'.format(wargs.val_prefix, wargs.val_src_suffix))
        val_trg_file = os.path.join(wargs.val_tst_dir, '{}.{}'.format(wargs.val_prefix, wargs.val_ref_suffix))
        wlog('\nPreparing validation set from {} and {} ... '.format(val_src_file, val_trg_file))
        valid_src_tlst, valid_trg_tlst = wrap_data(wargs.val_tst_dir, wargs.val_prefix,
                                                   wargs.val_src_suffix, wargs.val_ref_suffix,
                                                   src_vocab, trg_vocab, shuffle=False,
                                                   max_seq_len=wargs.dev_max_seq_len,
                                                   char=wargs.src_char)
        batch_valid = Input(valid_src_tlst, valid_trg_tlst, 1, batch_sort=False)

    batch_tests = None
    if wargs.tests_prefix is not None:
        assert isinstance(wargs.tests_prefix, list), 'Test files should be list.'
        init_dir(wargs.dir_tests)
        batch_tests = {}
        for prefix in wargs.tests_prefix:
            init_dir(wargs.dir_tests + '/' + prefix)
            test_file = '{}{}.{}'.format(wargs.val_tst_dir, prefix, wargs.val_src_suffix)
            wlog('\nPreparing test set from {} ... '.format(test_file))
            test_src_tlst, _ = wrap_tst_data(test_file, src_vocab, char=wargs.src_char)
            batch_tests[prefix] = Input(test_src_tlst, None, 1, batch_sort=False)
    wlog('\n## Finish to Prepare Dataset ! ##\n')

    src_emb = WordEmbedding(n_src_vcb, wargs.d_src_emb, wargs.input_dropout,
                            wargs.position_encoding, prefix='Src')
    trg_emb = WordEmbedding(n_trg_vcb, wargs.d_trg_emb, wargs.input_dropout,
                            wargs.position_encoding, prefix='Trg')
    # share the embedding matrix - preprocess with share_vocab required.
    if wargs.embs_share_weight:
        if n_src_vcb != n_trg_vcb:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')
        src_emb.we.weight = trg_emb.we.weight

    nmtModel = build_NMT(src_emb, trg_emb)

    if not wargs.copy_attn:
        classifier = Classifier(wargs.d_model if wargs.decoder_type == 'att' else 2 * wargs.d_enc_hid,
                                n_trg_vcb, trg_emb, loss_norm=wargs.loss_norm,
                                label_smoothing=wargs.label_smoothing,
                                emb_loss=wargs.emb_loss, bow_loss=wargs.bow_loss)
    nmtModel.decoder.classifier = classifier

    if wargs.gpu_id is not None:
        wlog('push model onto GPU {} ... '.format(wargs.gpu_id), 0)
        #nmtModel = nn.DataParallel(nmtModel, device_ids=wargs.gpu_id)
        nmtModel.to(tc.device('cuda'))
    else:
        wlog('push model onto CPU ... ', 0)
        nmtModel.to(tc.device('cpu'))
    wlog('done.')

    if wargs.pre_train is not None:
        assert os.path.exists(wargs.pre_train)
        from tools.utils import load_model
        _dict = load_model(wargs.pre_train)
        # initializing parameters of interactive attention model
        class_dict = None
        if len(_dict) == 5:
            model_dict, class_dict, eid, bid, optim = _dict
        elif len(_dict) == 4:
            model_dict, eid, bid, optim = _dict
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
            else: init_params(param, name, init_D=wargs.param_init_D, a=float(wargs.u_gain))

        wargs.start_epoch = eid + 1

    else:
        optim = Optim(wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm)
        #for n, p in nmtModel.named_parameters():
            # bias can not be initialized uniformly
            #if wargs.encoder_type != 'att' and wargs.decoder_type != 'att':
            #    init_params(p, n, init_D=wargs.param_init_D, a=float(wargs.u_gain))

    wlog(nmtModel)
    wlog(optim)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('parameters number: {}/{}'.format(pcnt1, pcnt2))

    wlog('\n' + '*' * 30 + ' trainable parameters ' + '*' * 30)
    for n, p in nmtModel.named_parameters():
        if p.requires_grad: wlog('{:60} : {}'.format(n, p.size()))

    optim.init_optimizer(nmtModel.parameters())

    trainer = Trainer(nmtModel, batch_train, vocabs, optim, batch_valid, batch_tests)

    trainer.train()


if __name__ == '__main__':

    main()















