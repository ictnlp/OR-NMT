# coding=utf-8
from __future__ import division

import os
import sys
import time
import argparse
import torch as tc
import torch.nn as nn
from torch import cuda

import sys
sys.path.append(os.getcwd())
import wargs

from tools.inputs import Input
from translate import Translator
from tools.utils import load_model, wlog, dec_conf, init_dir, append_file
from tools.inputs_handler import extract_vocab, wrap_tst_data
from models.losser import Classifier

if __name__ == '__main__':

    A = argparse.ArgumentParser(prog='NMT translator ... ')
    A.add_argument('-m', '--model-file', required=True, dest='model_file', help='model file')
    A.add_argument('-i', '--input-file', dest='input_file', default=None,
                   help='name of file to be translated')

    '''
    A.add_argument('--search-mode', dest='search_mode', default=2,
                   help='0: Greedy, 1&2: naive beam search, 3: cube pruning')

    A.add_argument('--beam-size', dest='beam_size', default=wargs.beam_size, help='beamsize')

    A.add_argument('--use-valid', dest='use_valid', type=int, default=0,
                   help='Translate valid set. (DEFAULT=0)')

    A.add_argument('--use-batch', dest='use_batch', type=int, default=0,
                   help='Whether we apply batch on beam search. (DEFAULT=0)')

    A.add_argument('--vocab-norm', dest='vocab_norm', type=int, default=1,
                   help='Whether we normalize the distribution of vocabulary (DEFAULT=1)')

    A.add_argument('--len-norm', dest='len_norm', type=int, default=1,
                   help='During searching, whether we normalize accumulated loss by length.')

    A.add_argument('--use-mv', dest='use_mv', type=int, default=0,
                   help='We use manipulation vacabulary by add this parameter. (DEFAULT=0)')

    A.add_argument('--merge-way', dest='merge_way', default='Him1',
                   help='merge way in cube pruning. (DEFAULT=s_im1. Him1/Hi/AiKL/LM)')

    A.add_argument('--avg-att', dest='avg_att', type=int, default=0,
                   help='Whether we average attention vector. (DEFAULT=0)')

    A.add_argument('--m-threshold', dest='m_threshold', type=float, default=0.,
                   help='a super-parameter to merge in cube pruning. (DEFAULT=0. no merge)')
    '''

    args = A.parse_args()
    model_file = args.model_file
    '''
    search_mode = args.search_mode
    beam_size = args.beam_size
    useValid = args.use_valid
    useBatch = args.use_batch
    vocabNorm = args.vocab_norm
    lenNorm = args.len_norm
    useMv = args.use_mv
    mergeWay = args.merge_way
    avgAtt = args.avg_att
    m_threshold = args.m_threshold
    switchs = [useBatch, vocabNorm, lenNorm, useMv, mergeWay, avgAtt]
    '''

    wlog('Starting load vocabularies ... ')
    assert os.path.exists(wargs.src_vcb) and os.path.exists(wargs.trg_vcb), 'need vocabulary ...'
    src_vocab = extract_vocab(None, wargs.src_vcb)
    trg_vocab = extract_vocab(None, wargs.trg_vcb)
    n_src_vcb, n_trg_vcb = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(n_src_vcb, n_trg_vcb))

    _dict = load_model(model_file)
    if len(_dict) == 4: model_dict, eid, bid, optim = _dict
    elif len(_dict) == 5: model_dict, class_dict, eid, bid, optim = _dict
    from models.embedding import WordEmbedding
    src_emb = WordEmbedding(n_src_vcb, wargs.d_src_emb, wargs.position_encoding, prefix='Src')
    trg_emb = WordEmbedding(n_trg_vcb, wargs.d_trg_emb, wargs.position_encoding, prefix='Trg')
    from models.model_builder import build_NMT
    nmtModel = build_NMT(src_emb, trg_emb)
    classifier = Classifier(wargs.d_dec_hid, n_trg_vcb, trg_emb, loss_norm=wargs.loss_norm,
                            label_smoothing=wargs.label_smoothing, emb_loss=wargs.emb_loss, bow_loss=wargs.bow_loss)
    nmtModel.decoder.classifier = classifier

    if wargs.gpu_id is not None:
        wlog('push model onto GPU {} ... '.format(wargs.gpu_id), 0)
        nmtModel.to(tc.device('cuda'))
    else:
        wlog('push model onto CPU ... ', 0)
        nmtModel.to(tc.device('cpu'))
    wlog('done.')
    nmtModel.load_state_dict(model_dict)
    wlog('\nFinish to load model.')

    dec_conf()

    nmtModel.eval()
    tor = Translator(nmtModel, src_vocab.idx2key, trg_vocab.idx2key, print_att=wargs.print_att)

    if not args.input_file:
        wlog('Translating one sentence ... ')
        # s = tc.Tensor([[0, 10811, 140, 217, 19, 1047, 482, 29999, 0, 0, 0]])
        #s = '( 北京 综合 电 ) 曾 因 美国 总统 布什 访华 而 一 度 升温 的 中 美 关系 在 急速 ' \
        #        '冷却 , 中国 昨天 证实 取消 今年 海军 舰艇 编队 访问 美国 港口 的 计划 , 并 ' \
        #        '拒绝 确定 国家 副主席 胡锦涛 是否 会 按 原定 计划 访美 。'
        #s = '黑夜 处处 有 , 神州 最 漫长 。'
        #s = '当 我 到 路口 时 我 这 边 的 灯 是 绿色 的 。'
        #s = '我 爱 北京 天安门 。'
        #s = '经过 国际 奥委会 的 不懈 努力 , 意大利 方面 在 冬奥会 开幕 前 四 天 作出 让步 , 承诺' \
        #' 冬奥会 期间 警方 不 会 进入 奥运村 搜查 运动员 驻地 , 但是 , 药检 呈 阳性 的 运动员 仍将'\
        #' 接受 意大利 检察 机关 的 调查 。'
        #s = '我 想 预约 一下 理发 。'
        #s = '玻利维亚 举行 总统 与 国会 选举 。'

        #s = '在 路易斯 安那 , 我 看见 一棵 松树 在 生长 , 它 独自 站 在 那里 , 枝条 上 挂着 苔藓 。'
        #t = 'I saw in Louisiana a live-oak growing, all alone stood it and the 3)moss hung down' \
        #    ' from the branches.'

        #s = "章启月 昨天 也 证实 了 俄罗斯 媒体 的 报道 , 说 中国 国家 主席 江泽民 前晚 应 " \
        #        "俄罗斯 总统 普京 的 要求 与 他 通 了 电话 , 双方 主要 是 就中 俄 互利 合作 " \
        #        "问题 交换 了 意见 。"
        #t = "( beijing , syndicated news ) the sino - us relation that was heated momentarily " \
        #        "by the us president bush 's visit to china is cooling down rapidly . china " \
        #        "confirmed yesterday that it has called off its naval fleet visit to the us " \
        #        "ports this year and refused to confirm whether the country 's vice president " \
        #        "hu jintao will visit the united states as planned ."

        s = '当 林肯 去 新奥尔良 时 , 我 听到 密西 西比 河 的 歌声 。'
        t = "When Lincoln goes to New Orleans, I hear Mississippi river's singing sound"
        #s = '新奥尔良 是 爵士 音乐 的 发源 地 。'
        #s = '新奥尔良 以 其 美食 而 闻名 。'
        # = '休斯顿 是 仅 次于 新奥尔良 和 纽约 的 美国 第三 大 港 。'

        s = [[src_vocab.key2idx[x] if x in src_vocab.key2idx else UNK for x in s.split(' ')]]
        t = [[trg_vocab.key2idx[x] if x in trg_vocab.key2idx else UNK for x in t.split(' ')]]
        tor.trans_samples(s, t)
        sys.exit(0)

    input_file = '{}{}.{}'.format(wargs.val_tst_dir, args.input_file, wargs.val_src_suffix)
    input_abspath = os.path.realpath(input_file)
    wlog('Translating test file {} ... '.format(input_abspath))
    ref_file = '{}{}.{}'.format(wargs.val_tst_dir, args.input_file, wargs.val_ref_suffix)
    test_src_tlst, _ = wrap_tst_data(input_abspath, src_vocab, char=wargs.src_char)
    test_input_data = Input(test_src_tlst, None, 1, batch_sort=False)

    batch_tst_data = None
    if os.path.exists(ref_file):
        wlog('With force decoding test file {} ... to get alignments'.format(input_file))
        wlog('\t\tRef file {}'.format(ref_file))
        from tools.inputs_handler import wrap_data
        tst_src_tlst, tst_trg_tlst = wrap_data(wargs.val_tst_dir, args.input_file,
                                               wargs.val_src_suffix, wargs.val_ref_suffix,
                                               src_vocab, trg_vocab, False, False, 1000000)
        batch_tst_data = Input(tst_src_tlst, tst_trg_tlst, 1, batch_sort=False)

    trans, alns = tor.single_trans_file(test_input_data, batch_tst_data=batch_tst_data)

    if wargs.search_mode == 0: p1 = 'greedy'
    elif wargs.search_mode == 1: p1 = 'nbs'
    elif wargs.search_mode == 2: p1 = 'cp'
    p2 = 'gpu' if wargs.gpu_id else 'cpu'
    p3 = 'wb' if wargs.with_batch else 'wob'

    outdir = 'wout_{}_{}_{}'.format(p1, p2, p3)
    if wargs.ori_search: outdir = '{}_{}'.format(outdir, 'ori')
    init_dir(outdir)
    outprefix = '{}/{}'.format(outdir, args.input_file)
    # wout_nbs_gpu_wb_wvalid/nist06_
    file_out = '{}_e{}_b{}_beam{}'.format(outprefix, eid, bid, wargs.beam_size)

    mteval_bleu = tor.write_file_eval(file_out, trans, args.input_file, alns)
    bleus_record_fname = '{}/record_bleu.log'.format(outdir)
    bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(eid, bid, mteval_bleu)
    with open(bleus_record_fname, 'a') as f:
        f.write(bleu_content + '\n')
        f.close()

    sfig = '{}/{}'.format(outdir, 'record_bleu.sfig')
    sfig_content = ('{} {} {} {} {}').format(
        #alpha,
        #beta,
        eid,
        bid,
        wargs.search_mode,
        wargs.beam_size,
        #kl,
        mteval_bleu
    )
    append_file(sfig, sfig_content)

