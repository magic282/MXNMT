import mxnet as mx

from ..rnn.LSTM import lstm, LSTMModel, LSTMParam, LSTMState
from ..rnn.GRU import gru, GRUModel, GRUParam, GRUState


class LstmDecoder(object):
    def __init__(self, seq_len, use_masking,
                 state_dim,
                 input_dim, output_dim,
                 vocab_size, embed_dim,
                 dropout=0.0, num_of_layer=1,
                 **kwargs):
        self.seq_len = seq_len
        self.use_masking = use_masking
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_of_layer = num_of_layer

    def decode(self, encoded):
        data = mx.sym.Variable('target')  # target input data
        label = mx.sym.Variable('target_softmax_label')  # target label data

        # declare variables
        embed_weight = mx.sym.Variable("target_embed_weight")
        cls_weight = mx.sym.Variable("target_cls_weight")
        cls_bias = mx.sym.Variable("target_cls_bias")
        init_weight = mx.sym.Variable("target_init_weight")
        init_bias = mx.sym.Variable("target_init_bias")
        input_weight = mx.sym.Variable("target_input_weight")
        input_bias = mx.sym.Variable("target_input_bias")

        param_cells = []
        last_states = []
        init_h = mx.sym.FullyConnected(data=encoded, num_hidden=self.state_dim * self.num_of_layer,
                                       weight=init_weight, bias=init_bias, name='init_fc')
        init_hs = mx.sym.SliceChannel(data=init_h, num_outputs=self.num_of_layer, squeeze_axis=1)
        for i in range(self.num_of_layer):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("target_l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("target_l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("target_l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("target_l%d_h2h_bias" % i)))
            # state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
            #                   h=mx.sym.Variable("target_l%d_init_h" % i))
            state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
                              h=init_hs[i])
            last_states.append(state)
        assert (len(last_states) == self.num_of_layer)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size + 1,
                                 weight=embed_weight, output_dim=self.embed_dim, name='target_embed')
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)
        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('target_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_target_mask')

        hidden_all = []
        for seq_idx in range(self.seq_len):
            con = mx.sym.Concat(wordvec[seq_idx], encoded)
            hidden = mx.sym.FullyConnected(data=con, num_hidden=self.embed_dim,
                                           weight=input_weight, bias=input_bias, name='input_fc')

            if self.use_masking:
                mask = masks[seq_idx]

            # stack LSTM
            for i in range(self.num_of_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                next_state = lstm(self.state_dim, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seq_idx, layeridx=i, dropout=dp_ratio)

                if self.use_masking:
                    prev_state_h = last_states[i].h
                    new_h = mx.sym.broadcast_mul(1.0 - mask, prev_state_h) + mx.sym.broadcast_mul(mask, next_state.h)
                    next_state = LSTMState(c=next_state.c, h=new_h)

                hidden = next_state.h
                last_states[i] = next_state
            # decoder
            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)

        hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.output_dim,
                                     weight=cls_weight, bias=cls_bias, name='target_pred')

        ################################################################################
        # Make label the same shape as our produced data path
        # I did not observe big speed difference between the following two ways

        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, shape=(-1,))
        if self.use_masking:
            loss_mask = mx.sym.transpose(data=input_mask)
            loss_mask = mx.sym.Reshape(data=loss_mask, shape=(-1, 1))
            pred = mx.sym.broadcast_mul(pred, loss_mask)

        # label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
        # label = [label_slice[t] for t in range(seq_len)]
        # label = mx.sym.Concat(*label, dim=0)
        # label = mx.sym.Reshape(data=label, target_shape=(0,))
        ################################################################################

        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='target_softmax')

        return sm


class LstmAttentionDecoder(object):
    def __init__(self, seq_len, use_masking,
                 state_dim,
                 input_dim, output_dim,
                 vocab_size, embed_dim,
                 dropout=0.0, num_of_layer=1,
                 attention=None, **kwargs):
        self.seq_len = seq_len
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

    def decode(self, encoded_for_init_state, encoded, encoded_mask):
        # last_encoded = encoded[-1]
        if self.attention:
            all_attended = mx.sym.Concat(*encoded, dim=1, name='concat_attended')  # (batch, n * seq_len)
            all_attended = mx.sym.Reshape(data=all_attended,
                                          shape=(self.kwargs['batch_size'], len(encoded), -1),
                                          name='_reshape_concat_attended')
        data = mx.sym.Variable('target')  # target input data
        label = mx.sym.Variable('target_softmax_label')  # target label data

        # declare variables
        embed_weight = mx.sym.Variable("target_embed_weight")
        cls_weight = mx.sym.Variable("target_cls_weight")
        cls_bias = mx.sym.Variable("target_cls_bias")
        init_weight = mx.sym.Variable("target_init_weight")
        init_bias = mx.sym.Variable("target_init_bias")
        # input_weight_W = mx.sym.Variable("target_input_W_weight")
        # input_weight_U = mx.sym.Variable("target_input_U_weight")
        input_weight = mx.sym.Variable("target_input_weight")
        # input_bias = mx.sym.Variable("target_input_bias")

        param_cells = []
        last_states = []
        init_h = mx.sym.FullyConnected(data=encoded_for_init_state, num_hidden=self.state_dim * self.num_of_layer,
                                       weight=init_weight, bias=init_bias, name='init_fc')
        init_h = mx.sym.Activation(data=init_h, act_type='tanh', name='init_act')
        init_hs = mx.sym.SliceChannel(data=init_h, num_outputs=self.num_of_layer, squeeze_axis=1)
        for i in range(self.num_of_layer):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("target_l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("target_l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("target_l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("target_l%d_h2h_bias" % i)))
            # state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
            #                   h=mx.sym.Variable("target_l%d_init_h" % i))
            state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
                              h=init_hs[i])
            last_states.append(state)
        assert (len(last_states) == self.num_of_layer)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size + 1,
                                 weight=embed_weight, output_dim=self.embed_dim, name='target_embed')
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)
        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('target_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_target_mask')

        hidden_all = []
        if self.attention:
            all_weights = []
        for seq_idx in range(self.seq_len):
            if self.attention:
                weights, weighted_encoded = self.attention.attend(attended=encoded, concat_attended=all_attended,
                                                                  state=last_states[0].h,
                                                                  attend_masks=encoded_mask,
                                                                  use_masking=True)
                all_weights.append(weights)
                hidden = weighted_encoded
            else:
                hidden = encoded

            # LSTM input x = W y_t-1 + U h
            hidden = mx.sym.Concat(wordvec[seq_idx], hidden)
            hidden = mx.sym.FullyConnected(data=hidden, num_hidden=self.state_dim,
                                           weight=input_weight, no_bias=True, name='input_fc')
            # hidden = mx.sym.dot(wordvec[seq_idx], input_weight_W) + mx.sym.dot(hidden, input_weight_U)

            if self.use_masking:
                mask = masks[seq_idx]

            # stack LSTM
            for i in range(self.num_of_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                next_state = lstm(self.state_dim, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seq_idx, layeridx=i, dropout=dp_ratio)

                if self.use_masking:
                    prev_state_h = last_states[i].h
                    prev_state_c = last_states[i].c
                    new_h = mx.sym.broadcast_mul(1.0 - mask, prev_state_h) + mx.sym.broadcast_mul(mask, next_state.h)
                    new_c = mx.sym.broadcast_mul(1.0 - mask, prev_state_c) + mx.sym.broadcast_mul(mask, next_state.c)
                    next_state = LSTMState(c=new_c, h=new_h)

                hidden = next_state.h
                last_states[i] = next_state
            # decoder
            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)

        hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.output_dim,
                                     weight=cls_weight, bias=cls_bias, name='target_pred')

        ################################################################################
        # Make label the same shape as our produced data path
        # I did not observe big speed difference between the following two ways

        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, shape=(-1,))
        # if self.use_masking:
        #     loss_mask = mx.sym.transpose(data=input_mask)
        #     loss_mask = mx.sym.Reshape(data=loss_mask, shape=(-1, 1))
        #     pred = mx.sym.broadcast_mul(pred, loss_mask)

        # label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
        # label = [label_slice[t] for t in range(seq_len)]
        # label = mx.sym.Concat(*label, dim=0)
        # label = mx.sym.Reshape(data=label, target_shape=(0,))
        ################################################################################

        sm = mx.sym.SoftmaxOutput(data=pred, label=label,
                                  use_ignore=True, ignore_label=0, normalization='valid',
                                  name='target_softmax')

        return sm
        # all_weights = [mx.sym.BlockGrad(data=w) for w in all_weights]
        # return mx.sym.Group([sm] + all_weights)


class GruAttentionDecoder(object):
    def __init__(self, seq_len, use_masking,
                 state_dim,
                 input_dim, output_dim,
                 vocab_size, embed_dim,
                 dropout=0.0, num_of_layer=1,
                 attention=None, **kwargs):
        self.seq_len = seq_len
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

    def decode(self, encoded_for_init_state, encoded, encoded_mask):
        # last_encoded = encoded[-1]
        if self.attention:
            all_attended = mx.sym.Concat(*encoded, dim=1, name='concat_attended')  # (batch, n * seq_len)
            all_attended = mx.sym.Reshape(data=all_attended,
                                          shape=(self.kwargs['batch_size'], len(encoded), -1),
                                          name='_reshape_concat_attended')
        data = mx.sym.Variable('target')  # target input data
        label = mx.sym.Variable('target_softmax_label')  # target label data

        # declare variables
        embed_weight = mx.sym.Variable("target_embed_weight")
        cls_weight = mx.sym.Variable("target_cls_weight")
        cls_bias = mx.sym.Variable("target_cls_bias")
        init_weight = mx.sym.Variable("target_init_weight")
        init_bias = mx.sym.Variable("target_init_bias")
        # input_weight_W = mx.sym.Variable("target_input_W_weight")
        # input_weight_U = mx.sym.Variable("target_input_U_weight")
        input_weight = mx.sym.Variable("target_input_weight")
        # input_bias = mx.sym.Variable("target_input_bias")

        param_cells = []
        last_states = []
        init_h = mx.sym.FullyConnected(data=encoded_for_init_state, num_hidden=self.state_dim * self.num_of_layer,
                                       weight=init_weight, bias=init_bias, name='init_fc')
        init_h = mx.sym.Activation(data=init_h, act_type='tanh', name='init_act')
        init_hs = mx.sym.SliceChannel(data=init_h, num_outputs=self.num_of_layer, squeeze_axis=1)
        for i in range(self.num_of_layer):
            param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable("target_l%d_i2h_gates_weight" % i),
                                        gates_i2h_bias=mx.sym.Variable("target_l%d_i2h_gates_bias" % i),
                                        gates_h2h_weight=mx.sym.Variable("target_l%d_h2h_gates_weight" % i),
                                        gates_h2h_bias=mx.sym.Variable("target_l%d_h2h_gates_bias" % i),
                                        trans_i2h_weight=mx.sym.Variable("target_l%d_i2h_trans_weight" % i),
                                        trans_i2h_bias=mx.sym.Variable("target_l%d_i2h_trans_bias" % i),
                                        trans_h2h_weight=mx.sym.Variable("target_l%d_h2h_trans_weight" % i),
                                        trans_h2h_bias=mx.sym.Variable("target_l%d_h2h_trans_bias" % i)))

            state = GRUState(h=init_hs[i])
            last_states.append(state)
        assert (len(last_states) == self.num_of_layer)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size + 1,
                                 weight=embed_weight, output_dim=self.embed_dim, name='target_embed')
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)
        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('target_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_target_mask')

        hidden_all = []
        if self.attention:
            all_weights = []
        for seq_idx in range(self.seq_len):
            if self.attention:
                weights, weighted_encoded = self.attention.attend(attended=encoded, concat_attended=all_attended,
                                                                  state=last_states[0].h,
                                                                  attend_masks=encoded_mask,
                                                                  use_masking=True)
                all_weights.append(weights)
                hidden = weighted_encoded
            else:
                hidden = encoded

            # GRU input x = W y_t-1 + U h
            hidden = mx.sym.Concat(wordvec[seq_idx], hidden)
            hidden = mx.sym.FullyConnected(data=hidden, num_hidden=self.state_dim,
                                           weight=input_weight, no_bias=True, name='input_fc')
            # hidden = mx.sym.dot(wordvec[seq_idx], input_weight_W) + mx.sym.dot(hidden, input_weight_U)

            if self.use_masking:
                mask = masks[seq_idx]

            # stack GRU
            for i in range(self.num_of_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                next_state = gru(self.state_dim, indata=hidden,
                                 prev_state=last_states[i],
                                 param=param_cells[i],
                                 seqidx=seq_idx, layeridx=i, dropout=dp_ratio)

                if self.use_masking:
                    prev_state_h = last_states[i].h
                    new_h = mx.sym.broadcast_mul(1.0 - mask, prev_state_h) + mx.sym.broadcast_mul(mask, next_state.h)
                    next_state = GRUState(h=new_h)

                hidden = next_state.h
                last_states[i] = next_state
            # decoder
            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)

        hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.output_dim,
                                     weight=cls_weight, bias=cls_bias, name='target_pred')

        ################################################################################
        # Make label the same shape as our produced data path
        # I did not observe big speed difference between the following two ways

        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, shape=(-1,))
        # if self.use_masking:
        #     loss_mask = mx.sym.transpose(data=input_mask)
        #     loss_mask = mx.sym.Reshape(data=loss_mask, shape=(-1, 1))
        #     pred = mx.sym.broadcast_mul(pred, loss_mask)

        # label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
        # label = [label_slice[t] for t in range(seq_len)]
        # label = mx.sym.Concat(*label, dim=0)
        # label = mx.sym.Reshape(data=label, target_shape=(0,))
        ################################################################################

        sm = mx.sym.SoftmaxOutput(data=pred, label=label,
                                  use_ignore=True, ignore_label=0, normalization='valid',
                                  name='target_softmax')

        return sm
        # all_weights = [mx.sym.BlockGrad(data=w) for w in all_weights]
        # return mx.sym.Group([sm] + all_weights)
