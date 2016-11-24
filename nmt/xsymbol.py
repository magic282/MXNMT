from mxwrap.seq2seq.encoder import BiDirectionalLstmEncoder
from mxwrap.seq2seq.decoder import LstmAttentionDecoder
from mxwrap.attention.BasicAttention import BasicAttention

import xconfig
import mxnet as mx


def s2s_unroll(s_num_lstm_layer, s_seq_len, s_vocab_size, s_num_hidden, s_num_embed, s_dropout,
               t_num_lstm_layer, t_seq_len, t_vocab_size, t_num_hidden, t_num_embed, t_num_label, t_dropout,
               **kwargs):
    encoder = BiDirectionalLstmEncoder(seq_len=s_seq_len, use_masking=True, state_dim=s_num_hidden,
                                       input_dim=s_vocab_size, output_dim=0,
                                       vocab_size=s_vocab_size, embed_dim=s_num_embed,
                                       dropout=s_dropout, num_of_layer=s_num_lstm_layer)

    attention = BasicAttention(batch_size=xconfig.batch_size, seq_len=s_seq_len, attend_dim=s_num_hidden * 2,
                               state_dim=t_num_hidden)

    decoder = LstmAttentionDecoder(seq_len=t_seq_len, use_masking=True, state_dim=t_num_hidden,
                                   input_dim=t_vocab_size, output_dim=t_num_label,
                                   vocab_size=t_vocab_size, embed_dim=t_num_embed, dropout=t_dropout,
                                   num_of_layer=t_num_lstm_layer, attention=attention, **kwargs)
    forward_hidden_all, backward_hidden_all, source_representations, source_mask_sliced = encoder.encode()

    encoded_for_init_state = mx.sym.Concat(forward_hidden_all[-1], backward_hidden_all[0], dim=1,
                                           name='encoded_for_init_state')
    target_representation = decoder.decode(encoded_for_init_state, source_representations, source_mask_sliced)
    return target_representation


def sym_gen(source_vocab_size, target_vocab_size):
    def _sym_gen(s_t_len):
        return s2s_unroll(s_num_lstm_layer=xconfig.num_lstm_layer, s_seq_len=s_t_len[0],
                          s_vocab_size=source_vocab_size + 1,
                          s_num_hidden=xconfig.num_hidden, s_num_embed=xconfig.num_embed, s_dropout=xconfig.dropout,
                          t_num_lstm_layer=xconfig.num_lstm_layer, t_seq_len=s_t_len[1],
                          t_vocab_size=target_vocab_size + 1,
                          t_num_hidden=xconfig.num_hidden, t_num_embed=xconfig.num_embed,
                          t_num_label=target_vocab_size + 1, t_dropout=xconfig.dropout, batch_size=xconfig.batch_size)

    return _sym_gen
