import math
import torch as tc
from torch.autograd import Variable

import wargs
from tools.bleu import *
from tools.utils import *
from tools.optimizer import Optim
from translate import Translator
from searchs.nbs import Nbs

class Trainer:

    def __init__(self, model, sv, tv, optim, trg_dict_size,
                 valid_data=None, tests_data=None, n_critic=1):

        self.lamda = 5
        self.eps = 1e-20
        #self.beta_KL = 0.005
        self.beta_KL = 0.
        self.beta_RLGen = 0.2
        self.clip_rate = 0.
        self.beta_RLBatch = 0.

        self.model = model
        self.decoder = model.decoder
        self.classifier = self.decoder.classifier
        self.sv, self.tv = sv, tv
        self.trg_dict_size = trg_dict_size

        self.n_critic = 1

        self.translator_sample = Translator(self.model, sv, tv, k=1, noise=False)
        #self.translator = Translator(model, sv, tv, k=10)
        if isinstance(optim, list):
            self.optim_G, self.optim_D = optim[0], optim[1]
            self.optim_G.init_optimizer(self.model.parameters())
            self.optim_D.init_optimizer(self.model.parameters())
        else:
            self.optim_G = Optim(
                'adam', 10e-05, wargs.max_grad_norm,
                learning_rate_decay=wargs.learning_rate_decay,
                start_decay_from=wargs.start_decay_from,
                last_valid_bleu=wargs.last_valid_bleu
            )
            self.optim_G.init_optimizer(self.model.parameters())
            self.optim_D = optim
            self.optim_D.init_optimizer(self.model.parameters())
            self.optim = [self.optim_G, self.optim_D]

        '''
        self.optim_RL = Optim(
            'adadelta', 1.0, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )
        self.optim_RL.init_optimizer(self.model.parameters())
        '''
        self.maskSoftmax = MaskSoftmax()
        self.valid_data = valid_data
        self.tests_data = tests_data

    def mt_eval(self, eid, bid, optim=None):

        if optim: self.optim = optim
        state_dict = { 'model': self.model.state_dict(), 'epoch': eid, 'batch': bid, 'optim': self.optim }

        if wargs.save_one_model: model_file = '{}.pt'.format(wargs.model_prefix)
        else: model_file = '{}_e{}_upd{}.pt'.format(wargs.model_prefix, eid, bid)
        tc.save(state_dict, model_file)
        wlog('Saving temporary model in {}'.format(model_file))

        self.model.eval()

        tor0 = Translator(self.model, self.sv, self.tv, print_att=wargs.print_att)
        BLEU = tor0.trans_eval(self.valid_data, eid, bid, model_file, self.tests_data)

        self.model.train()

        return BLEU

    # p1: (max_tlen_batch, batch_size, vocab_size)
    def distance(self, P, Q, y_masks, type='JS', y_gold=None):

        B = y_masks.size(1)
        hypo_N = y_masks.data.sum()

        if Q.size(0) > P.size(0): Q = Q[:(P.size(0) + 1)]

        if type == 'JS':
            #D_kl = tc.mean(tc.sum((tc.log(p1) - tc.log(p2)) * p1, dim=-1).squeeze(), dim=0)
            M = (P + Q) / 2.
            D_kl1 = tc.sum((tc.log(P) - tc.log(M)) * P, dim=-1).squeeze()
            D_kl2 = tc.sum((tc.log(Q) - tc.log(M)) * Q, dim=-1).squeeze()
            Js = 0.5 * D_kl1 + 0.5 * D_kl2
            sent_batch_dist = tc.sum(Js * y_masks) / B
            Js = Js / y_masks.sum(0)[None, :]
            word_level_dist = tc.sum(Js * y_masks) / B
            del M, D_kl1, D_kl2, Js

        elif type == 'KL':
            KL = tc.sum(P * (tc.log(P + self.eps) - tc.log(Q + self.eps)), dim=-1)
            # (L, B, V) -> (L, B)
            sent_batch_dist = tc.sum(KL * y_masks) / B
            word_level_dist0 = tc.sum(KL * y_masks) / hypo_N
            KL = KL / y_masks.sum(0)[None, :]
            #print W_KL.data
            word_level_dist1 = tc.sum(KL * y_masks) / B
            #print W_dist.data[0], y_masks.size(1)
            del KL

        elif type == 'KL-sent':

            #print p1[0]
            #print p2[0]
            #print '-----------------------------'
            p1 = tc.gather(p1, 2, y_gold[:, :, None])[:, :, 0]
            p2 = tc.gather(p2, 2, y_gold[:, :, None])[:, :, 0]
            # p1 (max_tlen_batch, batch_size)
            #print (p2 < 1) == False
            KL = (y_masks * (tc.log(p1) - tc.log(p2))) * p1
            sent_batch_dist = tc.sum(KL) / B
            KL = KL / y_masks.sum(0)[None, :]
            word_level_dist = tc.sum(KL * y_masks) / B
            # KL: (1, batch_size)
            del p1, p2, KL

        return sent_batch_dist, word_level_dist0, word_level_dist1

    def hyps_padding_dist(self, oracle, hyps_L, y_gold_maxL, p_y_hyp):

        #hyps_dist = [None] * B
        B, hyps_dist, hyps = oracle.size(1), [], [] # oracle, w/o bos
        assert (B == len(hyps_L)) and (oracle.size(0) == p_y_hyp.size(0))
        for bidx in range(B):
            hyp_L = hyps_L[bidx] - 1    # remove bos
            if hyp_L < y_gold_maxL:
                padding = tc.ones(y_gold_maxL - hyp_L) / self.trg_dict_size
                padding = padding[:, None].expand(padding.size(0), self.trg_dict_size)
                #pad = pad[:, None].expand((pad.size(0), one_p_y_hyp.size(-1)))
                padding = Variable(padding, requires_grad=False)
                if wargs.gpu_id and not padding.is_cuda: padding = padding.cuda()
                #print one_p_y_hyp.size(0), pad.size(0)
                #print tc.cat((p_y_hyp[:hyp_L, bidx, :], padding), dim=0).size()
                hyps_dist.append(tc.cat((p_y_hyp[:hyp_L, bidx, :], padding), dim=0))
                hyps.append(tc.cat((oracle[:hyp_L, bidx],
                                   Variable(PAD * tc.ones(y_gold_maxL - hyp_L).long()).cuda()), dim=0))
            else:
                hyps_dist.append(p_y_hyp[:y_gold_maxL, bidx, :])
                hyps.append(oracle[:y_gold_maxL, bidx])
            #hyps_dist[bidx] = one_p_y_hyp
        hyps_dist = tc.stack(hyps_dist, dim=1)
        hyps = tc.stack(hyps, dim=1)
        return hyps_dist, hyps

    def gumbel_sampling(self, B, y_maxL, feed_gold_out, noise=False):

        # feed_gold_out (L * B, V)
        logit = self.classifier.pred_map(feed_gold_out, noise=noise)

        if logit.is_cuda: logit = logit.cpu()
        hyps = tc.max(logit, 1)[1]
        # hyps (L*B, 1)
        hyps = hyps.view(y_maxL, B)
        hyps[0] = BOS * tc.ones(B).long()   # first words are <s>
        # hyps (L, B)
        c1 = tc.clamp((hyps.data - EOS), min=0, max=self.trg_dict_size)
        c2 = tc.clamp((EOS - hyps.data), min=0, max=self.trg_dict_size)
        _hyps = c1 + c2
        _hyps = tc.cat([_hyps, tc.zeros(B).long().unsqueeze(0)], 0)
        _hyps = tc.min(_hyps, 0)[1]
        #_hyps = tc.max(0 - _hyps, 0)[1]
        # idx: (1, B)
        hyps_L = _hyps.view(-1).tolist()
        hyps_mask = tc.zeros(y_maxL, B)
        for bid in range(B): hyps_mask[:, bid][:hyps_L[bid]] = 1.
        hyps_mask = Variable(hyps_mask, requires_grad=False)

        if wargs.gpu_id and not hyps_mask.is_cuda: hyps_mask = hyps_mask.cuda()
        if wargs.gpu_id and not hyps.is_cuda: hyps = hyps.cuda()

        return hyps, hyps_mask, hyps_L

    def try_trans(self, srcs, ref):

        # (len, 1)
        #src = sent_filter(list(srcs[:, bid].data))
        x_filter = sent_filter(list(srcs))
        y_filter = sent_filter(list(ref))
        #wlog('\n[{:3}] {}'.format('Src', idx2sent(x_filter, self.sv)))
        #wlog('[{:3}] {}'.format('Ref', idx2sent(y_filter, self.tv)))

        onebest, onebest_ids, _ = self.translator_sample.trans_onesent(x_filter)

        #wlog('[{:3}] {}'.format('Out', onebest))

        # no EOS and BOS
        return onebest_ids


    def beamsearch_sampling(self, srcs, trgs, eos=True):

        # y_masks: (trg_max_len, batch_size)
        B = srcs.size(1)
        oracles, oracles_L = [None] * B, [None] * B

        for bidx in range(B):
            onebest_ids = self.try_trans(srcs[:, bidx].data, trgs[:, bidx].data)

            if len(onebest_ids) == 0 or onebest_ids[0] != BOS: onebest_ids = [BOS] + onebest_ids
            if eos is True and onebest_ids[-1] != EOS: onebest_ids = onebest_ids + [EOS]
            oracles_L[bidx] = len(onebest_ids)
            oracles[bidx] = onebest_ids

        maxL = max(oracles_L)
        for bidx in range(B):
            cur_L, oracle = oracles_L[bidx], oracles[bidx]
            if cur_L < maxL: oracles[bidx] = oracle + [PAD] * (maxL - cur_L)

        oracles = Variable(tc.Tensor(oracles).long().t(), requires_grad=False) # -> (L, B)
        if wargs.gpu_id and not oracles.is_cuda: oracles = oracles.cuda()
        oracles_mask = oracles.ne(PAD).float()

        return oracles, oracles_mask, oracles_L

    def train(self, dh, dev_input, k, merge=False, name='default', percentage=0.1):

        #if (k + 1) % 1 == 0 and self.valid_data and self.tests_data:
        #    wlog('Evaluation on dev ... ')
        #    mt_eval(valid_data, self.model, self.sv, self.tv,
        #            0, 0, [self.optim, self.optim_RL, self.optim_G], self.tests_data)

        batch_count = len(dev_input)
        self.model.train()
        self.sampler = Nbs(self.model, self.tv, k=3, noise=False, print_att=False, batch_sample=True)

        for eid in range(wargs.start_epoch, wargs.max_epochs + 1):

            #self.optim_G.init_optimizer(self.model.parameters())
            #self.optim_RL.init_optimizer(self.model.parameters())

            size = int(percentage * batch_count)
            shuffled_batch_idx = tc.randperm(batch_count)

            wlog('{} NEW Epoch {}'.format('-' * 50, '-' * 50))
            wlog('{}, Epo:{:>2}/{:>2} start, random {}/{}({:.2%}) calc BLEU ... '.format(
                name, eid, wargs.max_epochs, size, batch_count, percentage), False)
            param_1, param_2, param_3, param_4, param_5, param_6 = [], [], [], [], [], []
            for k in range(size):
                bid, half_size = shuffled_batch_idx[k], wargs.batch_size

                # srcs: (max_sLen_batch, batch_size, emb), trgs: (max_tLen_batch, batch_size, emb)
                if merge is False: _, srcs, _, trgs, _, slens, srcs_m, trgs_m = dev_input[bid]
                else: _, srcs, _, trgs, _, slens, srcs_m, trgs_m = dh.merge_batch(dev_input[bid])[0]
                trgs, trgs_m = trgs[0], trgs_m[0]   # we only use the first dev reference

                if wargs.sampling == 'gumbeling':
                    oracles, oracles_mask, oracles_L = self.gumbel_sampling(B, y_gold_maxL, feed_gold_out, True)
                elif wargs.sampling == 'truncation':
                    oracles, oracles_mask, oracles_L = self.beamsearch_sampling(srcs, trgs)
                elif wargs.sampling == 'length_limit':
                    batch_beam_trgs = self.sampler.beam_search_trans(srcs, srcs_m, trgs_m)
                    hyps = [list(zip(*b)[0]) for b in batch_beam_trgs]
                    oracles = batch_search_oracle(hyps, trgs[1:], trgs_m[1:])
                    if wargs.gpu_id and not oracles.is_cuda: oracles = oracles.cuda()
                    oracles_mask = oracles.ne(0).float()
                    oracles_L = oracles_mask.sum(0).data.int().tolist()

                # oracles same with trgs, with bos and eos,(L, B)
                param_1.append(BLToStrList(oracles[1:-1].t(), [l-2 for l in oracles_L]))
                param_2.append(BLToStrList(trgs[1:-1].t(), trgs_m[1:-1].sum(0).data.int().tolist()))

                param_3.append(BLToStrList(oracles[1:-1, :half_size].t(),
                                       [l-2 for l in oracles_L[:half_size]]))
                param_4.append(BLToStrList(trgs[1:-1, :half_size].t(),
                                       trgs_m[1:-1, :half_size].sum(0).data.int().tolist()))
                param_5.append(BLToStrList(oracles[1:-1, half_size:].t(),
                                       [l-2 for l in oracles_L[half_size:]]))
                param_6.append(BLToStrList(trgs[1:-1, half_size:].t(),
                                       trgs_m[1:-1, half_size:].sum(0).data.int().tolist()))

            start_bat_bleu_hist = bleu('\n'.join(param_3), ['\n'.join(param_4)], logfun=debug)
            start_bat_bleu_new = bleu('\n'.join(param_5), ['\n'.join(param_6)], logfun=debug)
            start_bat_bleu = bleu('\n'.join(param_1), ['\n'.join(param_2)], logfun=debug)
            wlog('Random BLEU on history {}, new {}, mix {}'.format(
                start_bat_bleu_hist, start_bat_bleu_new, start_bat_bleu))

            wlog('Model selection and testing ... ')
            self.mt_eval(eid, 0, [self.optim_G, self.optim_D])
            if start_bat_bleu > 0.9:
                wlog('Better BLEU ... go to next data history ...')
                return

            s_kl_seen, w_kl_seen0, w_kl_seen1, rl_gen_seen, rl_rho_seen, rl_bat_seen, w_mle_seen, \
                    s_mle_seen, ppl_seen = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            for bid in range(batch_count):

                if merge is False: _, srcs, _, trgs, _, slens, srcs_m, trgs_m = dev_input[bid]
                else: _, srcs, _, trgs, _, slens, srcs_m, trgs_m = dh.merge_batch(dev_input[bid], True)[0]
                trgs, trgs_m = trgs[0], trgs_m[0]
                gold_feed, gold_feed_mask = trgs[:-1], trgs_m[:-1]
                gold, gold_mask = trgs[1:], trgs_m[1:]
                B, y_gold_maxL = srcs.size(1), gold_feed.size(0)
                N = gold.data.ne(PAD).sum()
                debug('B:{}, gold_feed_ymaxL:{}, N:{}'.format(B, y_gold_maxL, N))

                ###################################################################################
                debug('Optimizing KL distance ................................ {}'.format(name))
                #self.model.zero_grad()
                self.optim_G.zero_grad()

                feed_gold_out, _ = self.model(srcs, gold_feed, srcs_m, gold_feed_mask)
                p_y_gold = self.classifier.logit_to_prob(feed_gold_out)
                # p_y_gold: (gold_max_len - 1, B, trg_dict_size)

                if wargs.sampling == 'gumbeling':
                    oracles, oracles_mask, oracles_L = self.gumbel_sampling(B, y_gold_maxL, feed_gold_out, True)
                elif wargs.sampling == 'truncation':
                    oracles, oracles_mask, oracles_L = self.beamsearch_sampling(srcs, trgs)
                elif wargs.sampling == 'length_limit':
                    # w/o eos
                    batch_beam_trgs = self.sampler.beam_search_trans(srcs, srcs_m, trgs_m)
                    hyps = [list(zip(*b)[0]) for b in batch_beam_trgs]
                    oracles = batch_search_oracle(hyps, trgs[1:], trgs_m[1:])
                    if wargs.gpu_id and not oracles.is_cuda: oracles = oracles.cuda()
                    oracles_mask = oracles.ne(0).float()
                    oracles_L = oracles_mask.sum(0).data.int().tolist()

                oracle_feed, oracle_feed_mask = oracles[:-1], oracles_mask[:-1]
                oracle, oracle_mask = oracles[1:], oracles_mask[1:]
                # oracles same with trgs, with bos and eos,(L, B)
                feed_oracle_out, _ = self.model(srcs, oracle_feed, srcs_m, oracle_feed_mask)
                p_y_hyp = self.classifier.logit_to_prob(feed_oracle_out)
                p_y_hyp_pad, oracle = self.hyps_padding_dist(oracle, oracles_L, y_gold_maxL, p_y_hyp)
                #wlog('feed oracle dist: {}, feed gold dist: {}, oracle: {}'.format(p_y_hyp_pad.size(), p_y_gold.size(), oracle.size()))
                #B_KL_loss = self.distance(p_y_gold, p_y_hyp_pad, hyps_mask[1:], type='KL', y_gold=gold)
                S_KL_loss, W_KL_loss0, W_KL_loss1 = self.distance(
                    p_y_gold, p_y_hyp_pad, gold_mask, type='KL', y_gold=gold)
                debug('KL: Sent-level {}, Word0-level {}, Word1-level {}'.format(
                    S_KL_loss.data[0], W_KL_loss0.data[0], W_KL_loss1.data[0]))
                s_kl_seen += S_KL_loss.data[0]
                w_kl_seen0 += W_KL_loss0.data[0]
                w_kl_seen1 += W_KL_loss1.data[0]
                del p_y_hyp, feed_oracle_out

                ###################################################################################
                debug('Optimizing RL(Gen) .......... {}'.format(name))
                hyps_list = BLToStrList(oracle[:-1].t(), [l-2 for l in oracles_L], True)
                trgs_list = BLToStrList(trgs[1:-1].t(), trgs_m[1:-1].sum(0).data.int().tolist(), True)
                bleus_sampling = []
                for hyp, ref in zip(hyps_list, trgs_list):
                    bleus_sampling.append(bleu(hyp, [ref], logfun=debug))
                bleus_sampling = toVar(bleus_sampling, wargs.gpu_id)

                oracle_mask = oracle.ne(0).float()
                p_y_ahyp = p_y_hyp_pad.gather(2, oracle[:, :, None])[:, :, 0]
                p_y_ahyp = ((p_y_ahyp + self.eps).log() * oracle_mask).sum(0) / oracle_mask.sum(0)

                p_y_agold = p_y_gold.gather(2, gold[:, :, None])[:, :, 0]
                p_y_agold = ((p_y_agold + self.eps).log() * gold_mask).sum(0) / gold_mask.sum(0)

                r_theta = p_y_ahyp / p_y_agold
                A = 1. - bleus_sampling
                RL_Gen_loss = tc.min(r_theta * A, clip(r_theta, self.clip_rate) * A).sum()
                RL_Gen_loss = (RL_Gen_loss).div(B)
                debug('...... RL(Gen) cliped loss {}'.format(RL_Gen_loss.data[0]))
                rl_gen_seen += RL_Gen_loss.data[0]
                del p_y_agold

                ###################################################################################
                debug('Optimizing RL(Batch) -> Gap of MLE and BLEU ... rho ... feed onebest .... ')
                param_1 = BLToStrList(oracles[1:-1].t(), [l-2 for l in oracles_L])
                param_2 = BLToStrList(trgs[1:-1].t(), trgs_m[1:-1].sum(0).data.int().tolist())
                rl_bat_bleu = bleu(param_1, [param_2], logfun=debug)
                rl_avg_bleu = tc.mean(bleus_sampling).data[0]

                rl_rho = cor_coef(p_y_ahyp, bleus_sampling, eps=self.eps)
                rl_rho_seen += rl_rho.data[0]   # must use data, accumulating Variable needs more memory

                #p_y_hyp = p_y_hyp.exp()
                #p_y_hyp = (p_y_hyp * self.lamda / 3).exp()
                #p_y_hyp = self.maskSoftmax(p_y_hyp)
                p_y_ahyp = p_y_ahyp[None, :]
                p_y_ahyp_T = p_y_ahyp.t().expand(B, B)
                p_y_ahyp = p_y_ahyp.expand(B, B)
                p_y_ahyp_sum = p_y_ahyp_T + p_y_ahyp + self.eps

                #bleus_sampling = bleus_sampling[None, :].exp()
                bleus_sampling = self.maskSoftmax(self.lamda * bleus_sampling[None, :])
                bleus_T = bleus_sampling.t().expand(B, B)
                bleus = bleus_sampling.expand(B, B)
                bleus_sum = bleus_T + bleus + self.eps
                #print 'p_y_hyp_sum......................'
                #print p_y_hyp_sum.data
                RL_Batch_loss = p_y_ahyp / p_y_ahyp_sum * tc.log(bleus_T / bleus_sum) + \
                        p_y_ahyp_T / p_y_ahyp_sum * tc.log(bleus / bleus_sum)

                #RL_Batch_loss = tc.sum(-RL_Batch_loss * toVar(1 - tc.eye(B))).div(B)
                RL_Batch_loss = tc.sum(-RL_Batch_loss * toVar(1 - tc.eye(B), wargs.gpu_id))

                debug('RL(Batch) Mean BLEU: {}, rl_batch_loss: {}, rl_rho: {}, Bat BLEU: {}'.format(
                    rl_avg_bleu, RL_Batch_loss.data[0], rl_rho.data[0], rl_bat_bleu))
                rl_bat_seen += RL_Batch_loss.data[0]
                del oracles, oracles_mask, oracle_feed, oracle_feed_mask, oracle, oracle_mask,\
                        p_y_ahyp, bleus_sampling, bleus, p_y_ahyp_T, p_y_ahyp_sum, bleus_T, bleus_sum
                '''
                (self.beta_KL * S_KL_loss + self.beta_RLGen * RL_Gen_loss + \
                        self.beta_RLBatch * RL_Batch_loss).backward(retain_graph=True)

                mle_loss, grad_output, _ = memory_efficient(
                    feed_gold_out, gold, gold_mask, self.model.classifier)
                feed_gold_out.backward(grad_output)
                '''

                (self.beta_KL * W_KL_loss0 + self.beta_RLGen * RL_Gen_loss + \
                        self.beta_RLBatch * RL_Batch_loss).backward(retain_graph=True)
                self.optim_G.step()

                ###################################################### discrimitor
                #mle_loss, _, _ = self.classifier(feed_gold_out, gold, gold_mask)
                #mle_loss = mle_loss.div(B)
                #mle_loss = mle_loss.data[0]

                self.optim_D.zero_grad()
                mle_loss, _, _ = self.classifier.snip_back_prop(feed_gold_out, gold, gold_mask)
                self.optim_D.step()

                w_mle_seen += ( mle_loss / N )
                s_mle_seen += ( mle_loss / B )
                ppl_seen += math.exp(mle_loss/N)
                wlog('Epo:{:>2}/{:>2}, Bat:[{}/{}], W0-KL {:4.2f}, W1-KL {:4.2f}, '
                     'S-RLGen {:4.2f}, B-rho {:4.2f}, B-RLBat {:4.2f}, W-MLE:{:4.2f}, '
                     'S-MLE:{:4.2f}, W-ppl:{:4.2f}, B-bleu:{:4.2f}, A-bleu:{:4.2f}'.format(
                         eid, wargs.max_epochs, bid, batch_count, W_KL_loss0.data[0],
                         W_KL_loss1.data[0], RL_Gen_loss.data[0], rl_rho.data[0], RL_Batch_loss.data[0],
                         mle_loss/N, mle_loss/B, math.exp(mle_loss/N), rl_bat_bleu, rl_avg_bleu))
                #wlog('=' * 100)
                del S_KL_loss, W_KL_loss0, W_KL_loss1, RL_Gen_loss, RL_Batch_loss, feed_gold_out

            wlog('End epoch: S-KL {:4.2f}, W0-KL {:4.2f}, W1-KL {:4.2f}, S-RLGen {:4.2f}, B-rho '
                 '{:4.2f}, B-RLBat {:4.2f}, W-MLE {:4.2f}, S-MLE {:4.2f}, W-ppl {:4.2f}'.format(
                s_kl_seen/batch_count, w_kl_seen0/batch_count, w_kl_seen1/batch_count, rl_gen_seen/batch_count,
                rl_rho_seen/batch_count, rl_bat_seen/batch_count, w_mle_seen/batch_count,
                s_mle_seen/batch_count, ppl_seen/batch_count))

