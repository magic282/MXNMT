from mxwrap.seq2seq.encoder import BiDirectionalGruEncoder
from mxwrap.seq2seq.decoder import GruAttentionDecoder
from mxwrap.attention.ConcatAttention import ConcatAttention

import xconfig
import mxnet as mx


def s2s_unroll(encoder, attention, decoder,
               source_len, target_len,
               input_names, output_names,
               **kwargs):
    forward_hidden_all, backward_hidden_all, source_representations, source_mask_sliced = encoder.encode(source_len)

    encoded_for_init_state = mx.sym.Concat(forward_hidden_all[-1], backward_hidden_all[0], dim=1,
                                           name='encoded_for_init_state')
    target_representation = decoder.decode(target_len, encoded_for_init_state, source_representations,
                                           source_mask_sliced)
    return target_representation, input_names, output_names


def sym_gen(source_vocab_size, target_vocab_size):
    input_names = ['source', 'source_mask', 'target',
                   # 'target_mask',
                   "forward_source_l0_init_h",
                   "backward_source_l0_init_h"]
    output_names = ['target_softmax_label']
    encoder = BiDirectionalGruEncoder(use_masking=True, state_dim=xconfig.num_hidden,
                                      input_dim=0, output_dim=0,
                                      vocab_size=source_vocab_size, embed_dim=xconfig.num_embed,
                                      dropout=xconfig.dropout, num_of_layer=xconfig.num_lstm_layer)

    attention = ConcatAttention(batch_size=xconfig.batch_size, attend_dim=xconfig.num_hidden * 2,
                                state_dim=xconfig.num_hidden)

    decoder = GruAttentionDecoder(use_masking=True, state_dim=xconfig.num_hidden,
                                  input_dim=0, output_dim=target_vocab_size,
                                  vocab_size=target_vocab_size, embed_dim=xconfig.num_embed,
                                  dropout=xconfig.dropout,
                                  num_of_layer=xconfig.num_lstm_layer, attention=attention,
                                  batch_size=xconfig.batch_size)

    def _sym_gen(s_t_len):
        return s2s_unroll(encoder=encoder,
                          attention=attention,
                          decoder=decoder,
                          source_len=s_t_len[0],
                          target_len=s_t_len[1],
                          input_names=input_names, output_names=output_names,
                          )

    return _sym_gen
