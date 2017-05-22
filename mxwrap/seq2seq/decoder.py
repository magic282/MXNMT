import mxnet as mx

from ..rnn.GRU import GRU


class GruAttentionDecoder(object):
    def __init__(self, use_masking,
                 state_dim,
                 input_dim, output_dim,
                 vocab_size, embed_dim,
                 dropout=0.0, num_of_layer=1,
                 attention=None, **kwargs):
        self.use_masking = use_masking
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_of_layer = num_of_layer
        self.attention = attention
        self.kwargs = kwargs
        self.gru = GRU('decode', self.state_dim)
        # declare variables
        self.embed_weight = mx.sym.Variable("target_embed_weight")
        self.cls_weight = mx.sym.Variable("target_cls_weight")
        self.cls_bias = mx.sym.Variable("target_cls_bias")
        self.init_weight = mx.sym.Variable("target_init_weight")
        self.init_bias = mx.sym.Variable("target_init_bias")

    def decode(self, target_len, encoded_for_init_state, encoded, encoded_mask):
        # last_encoded = encoded[-1]

        data = mx.sym.Variable('target')  # target input data
        label = mx.sym.Variable('target_softmax_label')  # target label data

        hidden_all = [None for _ in range(target_len)]
        context_all = [None for _ in range(target_len)]
        all_weights = [None for _ in range(target_len)]
        readout_all = [None for _ in range(target_len)]

        init_h = mx.sym.FullyConnected(data=encoded_for_init_state, num_hidden=self.state_dim * self.num_of_layer,
                                       weight=self.init_weight, bias=self.init_bias, name='init_fc')
        init_h = mx.sym.Activation(data=init_h, act_type='tanh', name='init_act')

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                 weight=self.embed_weight, output_dim=self.embed_dim, name='target_embed')
        wordvec = mx.sym.split(data=embed, num_outputs=target_len, squeeze_axis=1)
        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('target_mask')
            # masks = mx.sym.split(data=input_mask, num_outputs=target_len, name='sliced_target_mask')

        source_attention_pre_compute = self.attention.pre_compute_fast(encoded)

        for seq_idx in range(target_len):
            # mask = masks[seq_idx] if self.use_masking else None
            if seq_idx == 0:
                hidden_all[seq_idx] = init_h
            else:
                in_x = mx.sym.Concat(wordvec[seq_idx], context_all[seq_idx - 1])
                # hidden_all[seq_idx] = self.gru.apply(in_x, hidden_all[seq_idx - 1], seq_idx, mask)
                hidden_all[seq_idx] = self.gru.apply(in_x, hidden_all[seq_idx - 1], seq_idx)

            weights, weighted_encoded = self.attention.attend_fast(source_pre_computed=source_attention_pre_compute,
                                                                   seq_len=len(encoded),
                                                                   state=hidden_all[seq_idx],
                                                                   attend_masks=encoded_mask,
                                                                   use_masking=True)
            context_all[seq_idx] = weighted_encoded
            all_weights[seq_idx] = weights
            readout_all[seq_idx] = mx.sym.Concat(wordvec[seq_idx], context_all[seq_idx], hidden_all[seq_idx])

        hidden_concat = mx.sym.Concat(*readout_all, dim=0)
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.output_dim,
                                     weight=self.cls_weight, bias=self.cls_bias, name='target_pred')

        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, shape=(-1,))

        sm = mx.sym.SoftmaxOutput(data=pred, label=label,
                                  use_ignore=True, ignore_label=0, normalization='valid',
                                  name='target_softmax')
        return sm

        # loss = mx.sym.softmax_cross_entropy(pred, label)
        # loss = mx.sym.MakeLoss(loss)
        # return loss
