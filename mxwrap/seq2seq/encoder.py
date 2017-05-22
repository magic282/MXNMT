import mxnet as mx

from ..rnn.GRU import  GRU


class BiDirectionalGruEncoder(object):
    def __init__(self, use_masking,
                 state_dim,
                 input_dim, output_dim,
                 vocab_size, embed_dim,
                 dropout=0.0, num_of_layer=1):
        self.use_masking = use_masking
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_of_layer = num_of_layer
        # declare variables
        self.forward_gru = GRU('forward_source', self.state_dim)
        self.backward_gru = GRU('backward_source', self.state_dim)
        self.embed_weight = mx.sym.Variable("source_embed_weight")

    def encode(self, seq_len):
        data = mx.sym.Variable('source')  # input data, source

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                 weight=self.embed_weight, output_dim=self.embed_dim, name='source_embed')
        wordvec = mx.sym.split(data=embed, num_outputs=seq_len, squeeze_axis=1)

        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('source_mask')
            enc_masks = mx.sym.split(data=input_mask, num_outputs=seq_len, squeeze_axis='False',
                                     name='sliced_source_mask')
            att_masks = mx.sym.split(data=input_mask, num_outputs=seq_len, squeeze_axis='True',
                                     name='sliced_source_mask')

        forward_hidden = [None for i in range(seq_len)]
        backward_hidden = [None for i in range(seq_len)]
        bi_hidden = []
        for seq_idx in range(seq_len):
            word = wordvec[seq_idx]
            mask = enc_masks[seq_idx] if self.use_masking else None
            if seq_idx == 0:
                forward_hidden[seq_idx] = mx.sym.Variable("forward_source_l0_init_h")
            else:
                forward_hidden[seq_idx] = self.forward_gru.apply(word, forward_hidden[seq_idx - 1], seq_idx, mask)

        for seq_idx in range(seq_len - 1, -1, -1):
            word = wordvec[seq_idx]
            mask = enc_masks[seq_idx] if self.use_masking else None
            if seq_idx == seq_len - 1:
                backward_hidden[seq_idx] = mx.sym.Variable("backward_source_l0_init_h")
            else:
                backward_hidden[seq_idx] = self.backward_gru.apply(word, backward_hidden[seq_idx + 1], seq_idx, mask)

        # for seq_idx in range(self.seq_len):
        for f, b in zip(forward_hidden, backward_hidden):
            bi = mx.sym.Concat(f, b, dim=1)
            bi_hidden.append(bi)

        if self.use_masking:
            return forward_hidden, backward_hidden, bi_hidden, att_masks
        else:
            return forward_hidden, backward_hidden, bi_hidden
