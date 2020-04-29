#( -*- coding: utf-8 -*-
from __future__ import division

import os
import re
import sys
import time
import numpy
import collections
from shutil import copyfile

import wargs
from tools.utils import *
from tools.mteval_bleu import mteval_bleu_file
from tools.multi_bleu import multi_bleu_file
numpy.set_printoptions(threshold=numpy.nan)

if wargs.decoder_type in ('rnn', 'gru', 'tgru'): from searchs.nbs import *
elif wargs.decoder_type == 'att': from searchs.nbs_t2t import *
if wargs.search_mode == 2: from searchs.cp import *

class Translator(object):

    def __init__(self, model, svcb_i2w=None, tvcb_i2w=None, search_mode=None, thresh=None, lm=None,
                 ngram=None, ptv=None, k=None, noise=None, print_att=False):

        self.svcb_i2w = svcb_i2w
        self.tvcb_i2w = tvcb_i2w
        self.lm = lm
        self.ptv = ptv
        self.noise = noise
        self.model = model
        self.ngram = ngram
        self.thresh = thresh
        self.print_att = print_att
        self.k = k if k else wargs.beam_size
        self.search_mode = search_mode if search_mode else wargs.search_mode

        if self.search_mode == 0: self.greedy = Greedy(self.tvcb_i2w)
        elif self.search_mode == 1: self.nbs = Nbs(model, self.tvcb_i2w, k=self.k,
                                                   noise=self.noise, print_att=print_att)
        elif self.search_mode == 2: self.wcp = Wcp(model, self.tvcb_i2w, k=self.k,
                                                   print_att=print_att)

    def trans_onesent(self, s):

        trans_start = time.time()

        with tc.no_grad():
            if self.search_mode == 0: trans = self.greedy.greedy_trans(s)
            elif self.search_mode == 1: batch_tran_cands = self.nbs.beam_search_trans(s)
            #elif self.search_mode == 2: (trans, ids), loss = self.wcp.cube_prune_trans(s)
            elif self.search_mode == 2: batch_tran_cands = self.wcp.cube_prune_trans(s)

        trans, loss, attent_matrix = batch_tran_cands[0][0] # first sent, best cand
        trans, ids, true_trans, true_idx, attent_matrix = filter_reidx(trans, self.tvcb_i2w, attent_matrix)

        #spend = time.time() - trans_start
        #wlog('Word-Level spend: {} / {} = {}'.format(
        #    format_time(spend), len(ids), format_time(spend / len(ids))))

        # attent_matrix: (trgL, srcL) numpy
        return trans, ids, true_trans, true_idx, attent_matrix

    def trans_samples(self, xs_nL, ys_nL):

        # xs_nL: (sample_size, max_sLen)
        for idx in range(len(xs_nL)):

            one_src = xs_nL[idx]
            x_filter = sent_filter(one_src)
            wlog('\n[{:3}] {}'.format('Ids', x_filter))
            x_sent = idx2sent(x_filter, self.svcb_i2w)
            if len(x_sent) == 2: x_sent, ori_src_toks = x_sent
            wlog('[{:3}] {}'.format('Src', x_sent))
            y_filter = sent_filter(ys_nL[idx])
            y_sent = idx2sent(y_filter, self.tvcb_i2w)
            if len(y_sent) == 2: y_sent, ori_ref_toks = y_sent
            wlog('[{:3}] {}'.format('Ref', y_sent))

            trans, ids, true_trans, _, attent_matrix = self.trans_onesent(one_src)

            src_toks = [] if x_sent == '' else x_sent.split(' ')
            trg_toks = [] if trans == '' else trans.split(' ')

            if wargs.with_bpe is True:
                wlog('[{:3}] {}'.format('Bpe', trans))
                trans = re.sub('(@@ )|(@@ ?$)', '', trans)

            if self.print_att is True:
                if isinstance(self.svcb_i2w, dict):
                    src_toks = [self.svcb_i2w[wid] for wid in x_filter]
                else:
                    src = self.svcb_i2w.decode(x_filter)
                    if len(src) == 2: x_sent, src_toks = src
                print_attention_text(attent_matrix, src_toks, trg_toks, isP=True)
                plot_attention(attent_matrix, src_toks, trg_toks, 'att.svg')

            wlog('[{:3}] {}'.format('Ori', true_trans))
            wlog('[{:3}] {}'.format('Out', trans))

    def force_decoding(self, batch_tst_data):

        batch_count = len(batch_tst_data)
        point_every, number_every = int(math.ceil(batch_count/100)), int(math.ceil(batch_count/10))
        attent_matrixs, trg_toks = [], []
        for batch_idx in range(batch_count):
            _, srcs, _, ttrgs_for_files, _, _, srcs_m, trg_mask_for_files = batch_tst_data[batch_idx]
            trgs, trgs_m = ttrgs_for_files[0], trg_mask_for_files[0]
            # src: ['我', '爱', '北京', '天安门']
            # trg: ['<b>', 'i', 'love', 'beijing', 'tiananmen', 'square', '<e>']
            # feed ['<b>', 'i', 'love', 'beijing', 'tiananmen', 'square']
            # must feed first <b>, because feed previous word to get alignment of next word 'i' !!!
            outputs = self.model(srcs, trgs[:-1], srcs_m, trgs_m[:-1], isAtt=True, test=True)
            if len(outputs) == 2: (_outputs, _checks) = outputs
            if len(_outputs) == 2: (outputs, attends) = _outputs
            else: attends = _checks
            # attends: (trg_maxL-1, src_maxL, B)

            attends = attends[:-1]
            attent_matrixs.extend(list(attends.permute(2, 0, 1).cpu().data.numpy()))
            # (trg_maxL-2, B) -> (B, trg_maxL-2)
            trg_toks.extend(list(trgs[1:-1].permute(1, 0).cpu().data.numpy()))

            if numpy.mod(batch_idx + 1, point_every) == 0: wlog('.', 0)
            if numpy.mod(batch_idx + 1, number_every) == 0: wlog('{}'.format(batch_idx + 1), 0)
        wlog('')

        assert len(attent_matrixs) == len(trg_toks)
        return attent_matrixs, trg_toks

    def single_trans_file(self, xs_inputs, src_labels_fname=None, batch_tst_data=None):

        n_batches = len(xs_inputs)   # number of batchs, here is sentences number
        point_every, number_every = int(math.ceil(n_batches/100)), int(math.ceil(n_batches/10))
        total_trans = []
        total_aligns = [] if self.print_att is True else None
        sent_no, words_cnt = 0, 0

        fd_attent_matrixs, trgs = None, None
        if batch_tst_data is not None:
            wlog('\nStarting force decoding ...')
            fd_attent_matrixs, trgs = self.force_decoding(batch_tst_data)
            wlog('Finish force decoding ...')

        trans_start = time.time()
        for bidx in range(n_batches):
            # (idxs, tsrcs, lengths, src_mask)
            xs_nL = xs_inputs[bidx][1]
            # idxs, tsrcs, ttrgs_for_files, lengths, src_mask, trg_mask_for_files
            for no in range(xs_nL.size(0)): # batch size, 1 for valid
                x_filter = sent_filter(xs_nL[no].tolist())
                if src_labels_fname is not None:
                    assert self.print_att is None, 'split sentence does not suport print attention'
                    # split by segment labels file
                    segs = self.segment_src(x_filter, labels[bidx].strip().split(' '))
                    trans = []
                    for seg in segs:
                        seg_trans, ids, _, _, _ = self.trans_onesent(seg)
                        words_cnt += len(ids)
                        trans.append(seg_trans)
                    # merge by order
                    trans = ' '.join(trans)
                else:
                    if fd_attent_matrixs is None:   # need translate
                        trans, ids, _, _, attent_matrix = self.trans_onesent(xs_nL[no].unsqueeze(0))
                        trg_toks = [] if trans == '' else trans.split(' ')
                        if trans == '': wlog('What ? null translation ... !')
                        words_cnt += len(ids)
                    else:
                        # attention: feed previous word -> get the alignment of next word !!!
                        attent_matrix = fd_attent_matrixs[bidx] # do not remove <b>
                        #print attent_matrix
                        trg_toks = sent_filter(trgs[bidx]) # remove <b> and <e>
                        trg_toks = [self.tvcb_i2w[wid] for wid in trg_toks]
                        trans = ' '.join(trg_toks)
                        words_cnt += len(trg_toks)

                    # get alignment from attent_matrix for one translation
                    if attent_matrix is not None:
                        # maybe generate null translation, fault-tolerant here
                        if isinstance(attent_matrix, list) and len(attent_matrix) == 0: alnStr = ''
                        else:
                            if isinstance(self.svcb_i2w, dict):
                                src_toks = [self.svcb_i2w[wid] for wid in x_filter]
                            else:
                                #print type(self.svcb_i2w)
                                # <class 'tools.text_encoder.SubwordTextEncoder'>
                                src_toks = self.svcb_i2w.decode(x_filter)
                                if len(src_toks) == 2: _, src_toks = src_toks
                                #print src_toks
                            # attent_matrix: (trgL, srcL) numpy
                            alnStr = print_attention_text(attent_matrix, src_toks, trg_toks)
                        total_aligns.append(alnStr)

                total_trans.append(trans)
                sent_no += 1
                if ( sent_no % point_every ) == 0:
                    wlog('.', newline=0)
                    sys.stderr.flush()
                if ( sent_no % number_every ) == 0: wlog(sent_no, newline=0)

        wlog(sent_no)

        if self.search_mode == 1:
            C = self.nbs.C
            if C[0] != 0:
                wlog('Average location of bp [{}/{}={:6.4f}]'.format(C[1], C[0], C[1] / C[0]))
                wlog('Step[{}] stepout[{}]'.format(*C[2:]))

        if self.search_mode == 2:
            C = self.wcp.C
            if C[0] != 0 and C[2] != 0:
                wlog('Average Merging Rate [{}/{}={:6.4f}]'.format(C[1], C[0], C[1] / C[0]))
                wlog('Average location of bp [{}/{}={:6.4f}]'.format(C[3], C[2], C[3] / C[2]))
                wlog('Step[{}] stepout[{}]'.format(*C[4:]))

        spend = time.time() - trans_start
        if words_cnt == 0: wlog('What ? No words generated when translating one file !!!')
        else:
            wlog('Word-Level spend: [{}/{} = {}/w], [{}/{:7.2f}s = {:7.2f} w/s]'.format(
                format_time(spend), words_cnt, format_time(spend / words_cnt),
                words_cnt, spend, words_cnt/spend))

        wlog('Done ...')
        if total_aligns is not None: total_aligns = '\n'.join(total_aligns) + '\n'
        return '\n'.join(total_trans) + '\n', total_aligns

    def segment_src(self, src_list, labels_list):

        #print len(src_list), len(labels_list)
        assert len(src_list) == len(labels_list)
        segments, seg = [], []
        for i in range(len(src_list)):
            c, l = src_list[i], labels_list[i]
            if l == 'S':
                segments.append([c])
            elif l == 'E':
                seg.append(c)
                segments.append(seg)
                seg = []
            elif l == 'B':
                if len(seg) > 0: segments.append(seg)
                seg = []
                seg.append(c)
            else:
                seg.append(c)

        return segments

    def write_file_eval(self, out_fname, trans, data_prefix, alns=None):

        if alns is not None:
            fout_aln = open('{}.aln'.format(out_fname), 'w')    # valids/trans
            fout_aln.writelines(alns)
            fout_aln.close()

        fout = open(out_fname, 'w')    # valids/trans
        fout.writelines(trans)
        fout.close()

        # *.ref
        ref_fpath = '{}{}.{}'.format(wargs.val_tst_dir, data_prefix, wargs.val_ref_suffix)
        ref_fpaths = grab_all_trg_files(ref_fpath)
        assert os.path.exists(out_fname), 'translation do not exist ...'
        if wargs.with_bpe is True:
            bpe_fname = '{}.bpe'.format(out_fname)
            wlog('copy {} to {} ... '.format(out_fname, bpe_fname), 0)
            #os.system('cp {} {}.bpe'.format(out_fname, out_fname))
            copyfile(out_fname, bpe_fname)
            assert os.path.exists(bpe_fname), 'bpe file do not exist ...'
            wlog('done')
            wlog("sed -r 's/(@@ )|(@@ ?$)//g' {} > {} ... ".format(bpe_fname, out_fname), 0)
            #os.system("sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(bpe_fname, out_fname))
            proc_bpe(bpe_fname, out_fname)
            wlog('done')

        if wargs.with_postproc is True:
            opost_name = '{}.opost'.format(out_fname)
            wlog('copy {} to {} ... '.format(out_fname, opost_name), 0)
            #os.system('cp {} {}'.format(out_fname, opost_name))
            copyfile(out_fname, opost_name)
            assert os.path.exists(opost_name), 'opost file do not exist ...'
            wlog('done')
            wlog("sh postproc.sh {} {}".format(opost_name, out_fname))
            os.system("sh postproc.sh {} {}".format(opost_name, out_fname))
            wlog('done')
            mteval_bleu_opost = mteval_bleu_file(opost_name, ref_fpaths, cased=wargs.cased)
            os.rename(opost_name, "{}_{}.txt".format(opost_name, mteval_bleu_opost))

        mteval_bleu = mteval_bleu_file(out_fname, ref_fpaths, cased=wargs.cased, char=wargs.char_bleu)
        multi_bleu = multi_bleu_file(out_fname, ref_fpaths, cased=wargs.cased, char=wargs.char_bleu)
        #mteval_bleu = mteval_bleu_file(out_fname + '.seg.plain', ref_fpaths)
        os.rename(out_fname, '{}{}_{}_{}.txt'.format(
            out_fname, '_char' if wargs.char_bleu is True else '', mteval_bleu, multi_bleu))

        final_bleu = multi_bleu if wargs.use_multi_bleu is True else mteval_bleu

        return final_bleu

    def trans_tests(self, tests_data, eid, bid):

        for _, test_prefix in zip(tests_data, wargs.tests_prefix):

            wlog('\nTranslating test dataset {}'.format(test_prefix))
            label_fname = '{}{}/{}.label'.format(wargs.val_tst_dir, wargs.seg_val_tst_dir,
                                                 test_prefix) if wargs.segments else None
            trans, alns = self.single_trans_file(tests_data[test_prefix], label_fname)

            outprefix = wargs.dir_tests + '/' + test_prefix + '/trans'
            test_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
                outprefix, eid, bid, self.k, self.search_mode, wargs.with_batch)

            _ = self.write_file_eval(test_out, trans, test_prefix, alns)

    def trans_eval(self, valid_data, eid, bid, model_file, tests_data):

        wlog('\nTranslating validation dataset {}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_src_suffix))
        label_fname = '{}{}/{}.label'.format(wargs.val_tst_dir, wargs.seg_val_tst_dir,
                                             wargs.val_prefix) if wargs.segments else None
        trans, alns = self.single_trans_file(valid_data, label_fname)

        outprefix = wargs.dir_valid + '/trans'
        valid_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
            outprefix, eid, bid, self.k, self.search_mode, wargs.with_batch)

        bleu_score = self.write_file_eval(valid_out, trans, wargs.val_prefix, alns)

        bleu_scores_fname = '{}/train_bleu.log'.format(wargs.dir_valid)
        bleu_scores = [0.]
        if os.path.exists(bleu_scores_fname):
            with open(bleu_scores_fname) as f:
                for line in f:
                    s_bleu = line.split(':')[-1].strip()
                    bleu_scores.append(float(s_bleu))

        wlog('\nCurrent [{}] - Best History [{}]'.format(bleu_score, max(bleu_scores)))
        if bleu_score > max(bleu_scores):   # better than history
            wargs.worse_counter = 0
            copyfile(model_file, wargs.best_model)
            wlog('Better, cp {} {}'.format(model_file, wargs.best_model))
            bleu_content = 'epoch [{}], batch[{}], BLEU score*: {}'.format(eid, bid, bleu_score)
            if tests_data is not None: self.trans_tests(tests_data, eid, bid)
        else:
            wlog('Worse')
            wargs.worse_counter = wargs.worse_counter + 1
            if wargs.worse_counter >= 8.:
                wlog('{} consecutive worses, finish training.'.format(wargs.worse_counter))
                sys.exit(0)
            bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(eid, bid, bleu_score)

        append_file(bleu_scores_fname, bleu_content)

        sfig = '{}.{}'.format(outprefix, 'sfig')
        sfig_content = ('{} {} {} {} {}').format(eid, bid, self.search_mode, self.k, bleu_score)
        append_file(sfig, sfig_content)

        if wargs.save_one_model and os.path.exists(model_file) is True:
            os.remove(model_file)
            wlog('Saving one model, so delete {}\n'.format(model_file))

        return bleu_score

if __name__ == "__main__":
    import sys
    res = valid_bleu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    wlog(res)




