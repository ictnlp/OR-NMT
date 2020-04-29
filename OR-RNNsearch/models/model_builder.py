import wargs
import torch.nn as nn


''' NMT model with encoder and decoder '''
class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, multigpu=False):

        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, src_mask=None, trg_mask=None, ss_prob=1.):
        '''
            src:        [batch_size, src_len]
            trg:        [batch_size, trg_len]
            src_mask:   [batch_size, src_len]
            trg_mask:   [batch_size, trg_len]
        Returns:
            * decoder output:   [trg_len, batch_size, hidden]
            * attention:        [batch_size, trg_len, src_len]
        '''

        contexts = None
        if wargs.encoder_type == 'gru':
            enc_output = self.encoder(src, src_mask)    # batch_size, max_L, hidden_size
            results = self.decoder(enc_output, trg, src_mask, trg_mask, ss_prob=ss_prob)
            logits, attends, contexts = results['logit'], results['attend'], results['context']
        if wargs.encoder_type == 'att':
            enc_output, _ = self.encoder(src, src_mask)
            logits, _, attends = self.decoder(trg, src, enc_output, trg_mask, src_mask)
            #logits, _, nlayer_attns = self.decoder(trg, src, enc_output)
            #attends = nlayer_attns[-1]
        elif wargs.encoder_type == 'tgru':
            enc_output = self.encoder(src, src_mask)    # batch_size, max_L, hidden_size
            results = self.decoder(enc_output, trg, src_mask, trg_mask, isAtt=True)
            if len(results) == 1: logits, attends = results, None
            elif len(results) == 2: logits, attends = results

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return {
            'logit': logits,
            'attend': attends,
            'context': contexts
        }


def build_encoder(src_emb):

    if wargs.encoder_type == 'gru':
        from models.gru_encoder import StackedGRUEncoder
        return StackedGRUEncoder(src_emb = src_emb,
                                 enc_hid_size = wargs.d_enc_hid,
                                 dropout_prob = wargs.rnn_dropout,
                                 n_layers = wargs.n_enc_layers)
    if wargs.encoder_type == 'att':
        from models.self_att_encoder import SelfAttEncoder
        return SelfAttEncoder(src_emb = src_emb,
                              n_layers = wargs.n_enc_layers,
                              d_model = wargs.d_model,
                              n_head = wargs.n_head,
                              d_ff_filter = wargs.d_ff_filter,
                              att_dropout = wargs.att_dropout,
                              residual_dropout = wargs.residual_dropout,
                              relu_dropout = wargs.relu_dropout,
                              encoder_normalize_before=wargs.encoder_normalize_before)
    elif wargs.encoder_type == 'cnn':
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif wargs.encoder_type == 'mean':
        return MeanEncoder(opt.enc_layers, embeddings)
    elif wargs.encoder_type == 'tgru':
        from models.tgru_encoder import StackedTransEncoder
        # 'Transition gru'
        return StackedTransEncoder(src_emb = src_emb,
                                   enc_hid_size = wargs.d_enc_hid,
                                   rnn_dropout = wargs.rnn_dropout,
                                   n_layers = wargs.n_enc_layers)

def build_decoder(trg_emb):

    if wargs.encoder_type == 'gru':
        from models.gru_decoder import StackedGRUDecoder
        return StackedGRUDecoder(trg_emb = trg_emb,
                                 enc_hid_size = wargs.d_enc_hid,
                                 dec_hid_size = wargs.d_dec_hid,
                                 n_layers = wargs.n_dec_layers,
                                 attention_type = wargs.attention_type,
                                 rnn_dropout_prob = wargs.rnn_dropout,
                                 out_dropout_prob = wargs.output_dropout)
    if wargs.decoder_type == 'att':
        from models.self_att_decoder import SelfAttDecoder
        return SelfAttDecoder(trg_emb = trg_emb,
                              n_layers = wargs.n_dec_layers,
                              d_model = wargs.d_model,
                              n_head = wargs.n_head,
                              d_ff_filter = wargs.d_ff_filter,
                              att_dropout = wargs.att_dropout,
                              residual_dropout = wargs.residual_dropout,
                              relu_dropout = wargs.relu_dropout,
                              proj_share_weight = wargs.proj_share_weight,
                              decoder_normalize_before = wargs.decoder_normalize_before)
    elif wargs.encoder_type == 'tgru':
        from models.tgru_decoder import StackedTransDecoder
        return StackedTransDecoder(trg_emb = trg_emb,
                                   enc_hid_size = wargs.d_enc_hid,
                                   dec_hid_size = wargs.d_dec_hid,
                                   n_head = wargs.n_head,
                                   n_layers = wargs.n_dec_layers,
                                   rnn_dropout = wargs.rnn_dropout,
                                   out_dropout_prob = wargs.output_dropout)

def build_NMT(src_emb, trg_emb):

    encoder = build_encoder(src_emb)
    decoder = build_decoder(trg_emb)

    nmt = NMTModel(encoder, decoder)

    return nmt


