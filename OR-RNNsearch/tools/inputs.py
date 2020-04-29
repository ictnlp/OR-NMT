from __future__ import division

import math
import wargs
import torch as tc
from .utils import *

class Input(object):

    def __init__(self, x_list, y_list, batch_size, batch_type='sents', bow=False,
                 batch_sort=False, prefix=None):

        self.bow = bow
        self.x_list = x_list
        self.n_sent = len(x_list)
        self.batch_size = batch_size
        self.batch_sort = batch_sort

        if y_list is not None:
            self.y_list_files = y_list
            # [sent0:[ref0, ref1, ...], sent1:[ref0, ref1, ... ], ...]
            assert self.n_sent == len(y_list)
            wlog('Bilingual: batch size {}, Sort in batch? {}'.format(self.batch_size, batch_sort))
        else:
            self.y_list_files = None
            wlog('Monolingual: batch size {}, Sort in batch? {}'.format(self.batch_size, batch_sort))

        self.prefix = prefix    # the prefix of data file, such as 'nist02' or 'nist03'
        self._read_pointer = 0
        if batch_type == 'sents':
            self.read_batch_fn = self.sents_batch
            self.n_batches = int(math.ceil(self.n_sent / self.batch_size))
        elif batch_type == 'token':
            self.read_batch_fn = self.token_batch
            self.n_batches, self._batch_pointer = 0, 0
        self.batch_type = batch_type

    def __len__(self):
        return self.n_batches

    def eos(self):
        end = ( self._read_pointer >= self.n_sent )
        #print '-----------', self._read_pointer, self.n_sent, end
        if end is True:
            self._read_pointer = 0
            if self.batch_type == 'token':
                self.n_batches = self._batch_pointer
                self._batch_pointer = 0
        return end

    def sents_batch(self, sent_idx, batch_size=80):

        x_batch_buffer = self.x_list[sent_idx * batch_size : (sent_idx + 1) * batch_size]
        y_batch_buffer = None
        if self.y_list_files is not None:
            # y_list_files: [sent_0:[ref0, ref1, ...], sent_1:[ref0, ref1, ... ], ...]
            y_batch_buffer = self.y_list_files[sent_idx * batch_size : (sent_idx + 1) * batch_size]

        return x_batch_buffer, y_batch_buffer

    def token_batch(self, sent_idx, batch_size=4096):

        max_len, n_samples, x_batch_buffer = 0, 0, []
        y_batch_buffer = None if self.y_list_files is None else []
        while max_len * n_samples < batch_size:     # may be greater than 4096
            #print self.n_sent, self._read_pointer
            if self._read_pointer == self.n_sent: break
            x = self.x_list[self._read_pointer]     # [[1,2,3,4,...]]
            x_batch_buffer.append(x)
            if self.y_list_files is not None:
                y = self.y_list_files[self._read_pointer]
                y_batch_buffer.append(y)
            #print max_len, n_samples, max_len * n_samples, batch_size, x
            #print y
            #print '=================='
            max_len = max(max_len, len(x[0]))
            n_samples += 1
            self._read_pointer += 1
        max_len, n_samples = 0, 0
        self._batch_pointer += 1

        return x_batch_buffer, y_batch_buffer

    def handle_batch(self, batch, bow=False):

        n_samples = len(batch)
        # [sent_0:[ref0, ref1, ...], sent_1:[ref0, ref1, ... ], ...]
        # -> [ref_0:[sent_0, sent_1, ...], ref_1:[sent_0, sent_1, ... ], ...]
        files_batch = [[one_sent_refs[ref_idx] for one_sent_refs in batch] \
                for ref_idx in range(len(batch[0]))]

        files_pad_batch, files_lens = [], []
        files_pad_batch_bow = [] if bow is True else None
        for a_file_batch in files_batch:   # a batch for one source/target file
            lens = []
            if bow is True: batch_bow_lists = []
            for sent_list in a_file_batch:
                lens.append(len(sent_list))
                if bow is True:
                    bag_of_words = list(set(sent_list))
                    if BOS in bag_of_words: bag_of_words.remove(BOS)    # do not include BOS in bow
                    batch_bow_lists.append(bag_of_words)
            _max_len = max(lens)
            if bow is True:
                bow_lens = [len(bow_list) for bow_list in batch_bow_lists]
                _max_bow_len = max(bow_lens)

            pad_batch = [[] for _ in range(n_samples)]
            if bow is True: pad_batch_bow = [[] for _ in range(n_samples)]
            for idx in range(n_samples):
                length = lens[idx]
                pad_batch[idx] = a_file_batch[idx] + [0] * (_max_len - length)
                if bow is True:
                    pad_batch_bow[idx] = batch_bow_lists[idx] + [0] * (_max_bow_len - bow_lens[idx])

            files_pad_batch.append(pad_batch)
            files_lens.append(lens)
            if files_pad_batch_bow is not None: files_pad_batch_bow.append(pad_batch_bow)

        return n_samples, files_pad_batch, files_lens, files_pad_batch_bow

    def __getitem__(self, idx):

        #assert idx < self.n_batches, 'idx:{} >= number of batches:{}'.format(idx, self.n_batches)
        src_batch, trg_batch = self.read_batch_fn(idx, self.batch_size)

        n_samples, srcs, slens, _ = self.handle_batch(src_batch)
        #wlog('this batch size {}'.format(n_samples))
        assert len(srcs) == 1, 'Requires only one in source side.'
        srcs, slens = srcs[0], slens[0]

        if self.y_list_files is not None:
            _, trgs_for_files, tlens_for_files, trg_bows_for_files = self.handle_batch(
                trg_batch, bow=self.bow)
            # -> [ref_0:[sent_0, sent_1, ...], ref_1:[sent_0, sent_1, ... ], ...]
            # trg_bows_for_files -> [ref_0:[bow_0, bow_1, ...], ref_1:[bow_0, bow_1, ... ], ...]

        # sort the source and target sentence
        idxs = range(n_samples)

        if self.batch_sort is True:
            if self.y_list_files is None:
                zipb = zip(idxs, srcs, slens)
                idxs, srcs, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
            else:
                # max length in different refs may differ, so can not tc.stack
                if trg_bows_for_files is None:
                    zipb = zip(idxs, srcs, zip(*trgs_for_files), slens)
                    idxs, srcs, trgs, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
                    #trgs_for_files = [tc.stack(ref) for ref in zip(*list(trgs))]
                else:
                    zipb = zip(idxs, srcs, zip(*trgs_for_files), zip(*trg_bows_for_files), slens)
                    idxs, srcs, trgs, trg_bows, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
                    trg_bows_for_files = [tc.LongTensor(ref_bow) for ref_bow in zip(*list(trg_bows))]
                trgs_for_files = [tc.tensor(ref).long() for ref in zip(*list(trgs))]

        lengths = tc.tensor(slens).int().view(1, -1)   # (1, batch_size)

        def tuple2Tenser(x):
            if x is None: return x
            # (batch_size, max_len)
            if isinstance(x, tuple) or isinstance(x, list): x = tc.tensor(x).long()
            if wargs.gpu_id is not None: x = x.cuda()    # push into GPU
            return x

        tsrcs = tuple2Tenser(srcs)
        src_mask = tsrcs.ne(0).float()

        if self.y_list_files is not None:

            ttrgs_for_files = [tuple2Tenser(trgs) for trgs in trgs_for_files]
            trg_mask_for_files = [ttrgs.ne(0).float() for ttrgs in ttrgs_for_files]
            if trg_bows_for_files is not None:
                ttrg_bows_for_files = [tuple2Tenser(trg_bows) for trg_bows in trg_bows_for_files]
                ttrg_bows_mask_for_files = [ttrg_bows.ne(0).float() for ttrg_bows in ttrg_bows_for_files]
            else: ttrg_bows_for_files, ttrg_bows_mask_for_files = None, None

            '''
                [list] idxs: sorted idx by ascending order of source lengths in one batch
                [tensor] tsrcs: padded source batch, tensor(batch_size, max_len)
                [list] ttrgs_for_files: list of tensors (padded target batch),
                            [tensor(batch_size, max_len), ..., ]
                            each item in this list for one target reference file one batch
                [intTensor] lengths: sorted source lengths by ascending order, (1, batch_size)
                [tensor] src_mask: 0/1 tensor(0 for padding) (batch_size, max_len)
                [list] trg_mask_for_files: list of 0/1 Variables (0 for padding)
                            [tensor(batch_size, max_len), ..., ]
                            each item in this list for one target reference file one batch
            '''
            return idxs, tsrcs, ttrgs_for_files, ttrg_bows_for_files, lengths, \
                    src_mask, trg_mask_for_files, ttrg_bows_mask_for_files

        else:

            return idxs, tsrcs, lengths, src_mask


    def shuffle(self):

        wlog('shuffling the whole training data bilingually ... ', False)
        rand_idxs = tc.randperm(self.n_sent).tolist()
        self.x_list = [self.x_list[k] for k in rand_idxs]
        self.y_list_files = [self.y_list_files[k] for k in rand_idxs]
        #data = list(zip(self.x_list, self.y_list_files))
        #x_tuple, y_tuple = zip(*[data[i] for i in tc.randperm(len(data))])
        #self.x_list, self.y_list_files = list(x_tuple), list(y_tuple)
        wlog('done.')

        slens = [len(self.x_list[k]) for k in range(self.n_sent)]
        self.x_list, self.y_list_files = sort_batches(self.x_list, self.y_list_files,
                                                      slens, wargs.batch_size,
                                                      wargs.sort_k_batches)


