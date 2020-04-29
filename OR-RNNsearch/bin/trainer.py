from __future__ import division, print_function

import os
import sys
import math
import time
import random
import subprocess

import numpy as np
import torch as tc

import wargs
from tools.utils import *
from searchs.nbs import Nbs
from translate import Translator

class Trainer(object):

    def __init__(self, model, train_data, vocab_data, optim, valid_data=None, tests_data=None):

        self.model, self.classifier, self.optim = model, model.decoder.classifier, optim
        self.sv, self.tv = vocab_data['src'].idx2key, vocab_data['trg'].idx2key
        self.train_data, self.valid_data, self.tests_data = train_data, valid_data, tests_data
        self.max_epochs, self.start_epoch = wargs.max_epochs, wargs.start_epoch

        wlog('self-normalization alpha -> {}'.format(wargs.self_norm_alpha))

        self.n_look = wargs.n_look
        assert self.n_look <= wargs.batch_size, 'eyeball count > batch size'
        self.n_batches = len(train_data)    # [low, high)

        self.look_xs, self.look_ys = None, None
        if wargs.fix_looking is True:
            rand_idxs = random.sample(range(train_data.n_sent), self.n_look)
            wlog('randomly look {} samples frow the whole training data'.format(self.n_look))
            self.look_xs = [train_data.x_list[i][0] for i in rand_idxs]
            self.look_ys = [train_data.y_list_files[i][0] for i in rand_idxs]
        self.look_tor = Translator(self.model, self.sv, self.tv)
        self.n_eval = 1
        self.tor = Translator(self.model, self.sv, self.tv, print_att=wargs.print_att)

        self.snip_size, self.trunc_size = wargs.snip_size, wargs.trunc_size
        self.grad_accum_count = wargs.grad_accum_count

        self.epoch_shuffle_train = wargs.epoch_shuffle_train
        self.epoch_shuffle_batch = wargs.epoch_shuffle_batch
        self.ss_cur_prob = wargs.ss_prob_begin
        if wargs.ss_type is not None:
            wlog('word-level optimizing bias between training and decoding ...')
            if wargs.bleu_sampling is True: wlog('sentence-level optimizing ...')
            wlog('schedule sampling value {}'.format(self.ss_cur_prob))
            if self.ss_cur_prob < 1. and wargs.bleu_sampling is True:
                self.sampler = Nbs(self.model, self.tv, k=3, noise=wargs.bleu_gumbel_noise,
                                   batch_sample=True)
        if self.grad_accum_count > 1:
            assert(self.trunc_size == 0), 'to accumulate grads, disable target sequence truncating'

    def accum_matrics(self, batch_size, xtoks, ytoks, nll, ok_ytoks, logZ):

        self.look_sents += batch_size
        self.e_sents += batch_size
        self.look_nll += nll
        self.look_ok_ytoks += ok_ytoks
        self.e_nll += nll
        self.e_ok_ytoks += ok_ytoks
        self.look_xtoks += xtoks
        self.look_ytoks += ytoks
        self.e_ytoks += ytoks
        self.look_batch_logZ += logZ
        self.e_batch_logZ += logZ

    def grad_accumulate(self, real_batches, epo):

        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in real_batches:

            # (batch_size, max_slen_batch)
            _, xs, y_for_files, bows, x_lens, xs_mask, y_mask_for_files, bows_mask = batch
            _batch_size = xs.size(0)
            ys, ys_mask = y_for_files[0], y_mask_for_files[0]
            #wlog('x: {}, x_mask: {}, y: {}, y_mask: {}'.format(
            #    xs.size(), xs_mask.size(), ys.size(), ys_mask.size()))
            if bows is not None:
                bows, bows_mask = bows[0], bows_mask[0]
                #wlog('bows: {}, bows_mask: {})'.format(bows.size(), bows_mask.size()))
            _xtoks = xs.data.ne(PAD).sum().item()
            assert _xtoks == x_lens.data.sum().item()
            _ytoks = ys[1:].data.ne(PAD).sum().item()

            ys_len = ys.size(1)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else ys_len

            for j in range(0, ys_len - 1, trunc_size):
                # 1. Create truncated target.
                part_ys = ys[:, j : j + trunc_size]
                part_ys_mask = ys_mask[:, j : j + trunc_size]
                # 2. F-prop all but generator.
                if self.grad_accum_count == 1: self.model.zero_grad()
                # exclude last target word from inputs
                results = self.model(xs, part_ys[:, :-1], xs_mask, part_ys_mask[:, :-1], self.ss_cur_prob)
                logits, alphas, contexts = results['logit'], results['attend'], results['context']
                # (batch_size, y_Lm1, out_size)

                gold, gold_mask = part_ys[:, 1:].contiguous(), part_ys_mask[:, 1:].contiguous()
                # 3. Compute loss in shards for memory efficiency.
                _nll, _ok_ytoks, _logZ = self.classifier.snip_back_prop(
                    logits, gold, gold_mask, bows, bows_mask, epo, self.snip_size, contexts)

            self.accum_matrics(_batch_size, _xtoks, _ytoks, _nll, _ok_ytoks, _logZ)
        # 3. Update the parameters and statistics.
        self.optim.step()
        #tc.cuda.empty_cache()

    def look_samples(self, bidx):

        if bidx % wargs.look_freq == 0:

            look_start = time.time()
            self.model.eval()   # affect the dropout !!!
            if self.look_xs is not None and self.look_ys is not None:
                _xs, _ys = self.look_xs, self.look_ys
            else:
                rand_idxs = random.sample(range(self.train_data.n_sent), self.n_look)
                wlog('randomly look {} samples frow the whole training data'.format(self.n_look))
                _xs = [self.train_data.x_list[i][0] for i in rand_idxs]
                _ys = [self.train_data.y_list_files[i][0] for i in rand_idxs]
            self.look_tor.trans_samples(_xs, _ys)
            wlog('')
            self.look_spend = time.time() - look_start
            self.model.train()

    def try_valid(self, epo, e_bidx, n_steps):

        if wargs.epoch_eval is not True and n_steps > wargs.eval_valid_from and \
           n_steps % wargs.eval_valid_freq == 0:
            eval_start = time.time()
            wlog('\nAmong epoch, batch [{}], {}-th validation ...'.format(e_bidx, self.n_eval))
            self.mt_eval(epo, e_bidx)
            self.eval_spend = time.time() - eval_start

    def mt_eval(self, eid, bid):

        state_dict = { 'model': self.model.state_dict(), 'epoch': eid, 'batch': bid, 'optim': self.optim }

        if wargs.save_one_model: model_file = '{}.pt'.format(wargs.model_prefix)
        else: model_file = '{}_e{}_upd{}.pt'.format(wargs.model_prefix, eid, bid)
        tc.save(state_dict, model_file)
        wlog('Saving temporary model in {}'.format(model_file))

        self.model.eval()
        self.tor.trans_eval(self.valid_data, eid, bid, model_file, self.tests_data)
        self.model.train()
        self.n_eval += 1

    def train(self):

        wlog('start training ... ')
        train_start = time.time()
        wlog('\n' + '#' * 120 + '\n' + '#' * 30 + ' Start Training ' + '#' * 30 + '\n' + '#' * 120)
        batch_oracles, _checks, accum_batches, real_batches = None, None, 0, []
        self.model.train()

        for epo in range(self.start_epoch, self.max_epochs + 1):

            wlog('\n{} Epoch [{}/{}] {}'.format('$'*30, epo, self.max_epochs, '$'*30))
            # shuffle the training data for each epoch
            if self.epoch_shuffle_train: self.train_data.shuffle()
            self.e_nll, self.e_ytoks, self.e_ok_ytoks, self.e_batch_logZ, self.e_sents \
                    = 0, 0, 0, 0, 0
            self.look_nll, self.look_ytoks, self.look_ok_ytoks, self.look_batch_logZ, \
                    self.look_sents = 0, 0, 0, 0, 0
            self.look_xtoks, self.look_spend, b_counter, eval_spend = 0, 0, 0, 0
            epo_start = show_start = time.time()
            if self.epoch_shuffle_batch: shuffled_bidx = tc.randperm(self.n_batches)

            #for bidx in range(self.n_batches):
            bidx = 0
            cond = True if wargs.lr_update_way != 'invsqrt' else self.optim.learning_rate > wargs.min_lr
            while cond:
                if self.train_data.eos() is True: break
                b_counter += 1
                e_bidx = shuffled_bidx[bidx] if self.epoch_shuffle_batch else bidx
                if wargs.ss_type is not None and self.ss_cur_prob < 1. and wargs.bleu_sampling:
                    batch_beam_trgs = self.sampler.beam_search_trans(xs, xs_mask, ys_mask)
                    batch_beam_trgs = [list(zip(*b)[0]) for b in batch_beam_trgs]
                    #wlog(batch_beam_trgs)
                    batch_oracles = batch_search_oracle(batch_beam_trgs, ys[1:], ys_mask[1:])
                    #wlog(batch_oracles)
                    batch_oracles = batch_oracles[:-1].cuda()
                    batch_oracles = self.model.decoder.trg_lookup_table(batch_oracles)

                batch = self.train_data[e_bidx]
                real_batches.append(batch)
                accum_batches += 1
                if accum_batches == self.grad_accum_count:

                    self.grad_accumulate(real_batches, epo)
                    current_steps = self.optim.n_current_steps
                    accum_batches, real_batches = 0, []
                    grad_checker(self.model, _checks)
                    if current_steps % wargs.display_freq == 0:
                        #print self.look_ok_ytoks, self.look_nll, self.look_ytoks, self.look_nll/self.look_ytoks
                        ud = time.time() - show_start - self.look_spend - eval_spend
                        wlog(
                            'Epo:{:>2}/{:>2} |[{:^5}/{} {:^5}] |acc:{:5.2f}% |nll:{:4.2f}'
                            ' |w-ppl:{:4.2f} |w(s)-logZ|:{:.2f}({:.2f}) '
                            ' |x(y)/s:{:>4}({:>4})/{}={}({}) |x(y)/sec:{}({}) |lr:{:7.6f}'
                            ' |{:4.2f}s/{:4.2f}m'.format(
                                epo, self.max_epochs, b_counter, len(self.train_data), current_steps,
                                (self.look_ok_ytoks / self.look_ytoks) * 100,
                                self.look_nll / self.look_ytoks,
                                math.exp(self.look_nll / self.look_ytoks),
                                self.look_batch_logZ / self.look_ytoks,
                                self.look_batch_logZ / self.look_sents, self.look_xtoks,
                                self.look_ytoks, self.look_sents,
                                int(round(self.look_xtoks / self.look_sents)),
                                int(round(self.look_ytoks / self.look_sents)),
                                int(round(self.look_xtoks / ud)), int(round(self.look_ytoks / ud)),
                                self.optim.learning_rate, ud, (time.time() - train_start) / 60.)
                        )
                        self.look_nll, self.look_xtoks, self.look_ytoks, self.look_ok_ytoks, \
                                self.look_batch_logZ, self.look_sents = 0, 0, 0, 0, 0, 0
                        self.look_spend, eval_spend = 0, 0
                        show_start = time.time()

                    self.look_samples(current_steps)
                    self.try_valid(epo, e_bidx, current_steps)
                bidx += 1

            avg_epo_acc, avg_epo_nll = self.e_ok_ytoks/self.e_ytoks, self.e_nll/self.e_ytoks
            wlog('\nEnd epoch [{}]'.format(epo))
            wlog('avg. w-acc: {:4.2f}%, w-nll: {:4.2f}, w-ppl: {:4.2f}'.format(
                avg_epo_acc * 100, avg_epo_nll, math.exp(avg_epo_nll)))
            wlog('avg. |w-logZ|: {:.2f}/{}={:.2f} |s-logZ|: {:.2f}/{}={:.2f}'.format(
                self.e_batch_logZ, self.e_ytoks, self.e_batch_logZ / self.e_ytoks,
                self.e_batch_logZ, self.e_sents, self.e_batch_logZ / self.e_sents))
            if wargs.epoch_eval is True:
                wlog('\nEnd epoch, batch [{}], {}-th validation ...'.format(e_bidx, self.n_eval))
                self.mt_eval(epo, e_bidx)
            # decay the probability value epslion of scheduled sampling per batch
            if wargs.ss_type is not None: self.ss_cur_prob = ss_prob_decay(epo)   # start from 1.
            epo_time_consume = time.time() - epo_start
            wlog('Consuming: {:4.2f}s'.format(epo_time_consume))

        wlog('Finish training, comsuming {:6.2f} hours'.format((time.time() - train_start) / 3600))
        wlog('Congratulations!')

