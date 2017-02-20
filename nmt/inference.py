import mxnet as mx
from mxwrap.rnn.LSTM import lstm, LSTMModel, LSTMParam, LSTMState
from mxwrap.seq2seq.encoder import LstmEncoder, BiDirectionalLstmEncoder
from mxwrap.attention.BasicAttention import BasicAttention


def initial_state_symbol(t_num_lstm_layer, t_num_hidden):
    encoded = mx.sym.Variable("encoded")
    init_weight = mx.sym.Variable("target_init_weight")
    init_bias = mx.sym.Variable("target_init_bias")
    init_h = mx.sym.FullyConnected(data=encoded, num_hidden=t_num_hidden * t_num_lstm_layer,
                                   weight=init_weight, bias=init_bias, name='init_fc')
    init_h = mx.sym.Activation(data=init_h, act_type='tanh', name='init_act')
    init_hs = mx.sym.SliceChannel(data=init_h, num_outputs=t_num_lstm_layer, squeeze_axis=1)
    return init_hs


class BiS2SInferenceModel(object):
    def __init__(self,
                 s_num_lstm_layer, s_seq_len, s_vocab_size, s_num_hidden, s_num_embed, s_dropout,
                 t_num_lstm_layer, t_seq_len, t_vocab_size, t_num_hidden, t_num_embed, t_num_label, t_dropout,
                 arg_params,
                 use_masking,
                 ctx=mx.cpu(),
                 batch_size=1):
        self.encode_sym = bidirectional_encode_symbol(s_num_lstm_layer, s_seq_len, use_masking,
                                                      s_vocab_size, s_num_hidden, s_num_embed,
                                                      s_dropout)
        attention = BasicAttention(batch_size=batch_size, seq_len=s_seq_len, attend_dim=s_num_hidden * 2,
                                   state_dim=t_num_hidden)
        self.decode_sym = lstm_attention_decode_symbol(t_num_lstm_layer, t_seq_len, t_vocab_size, t_num_hidden,
                                                       t_num_embed,
                                                       t_num_label, t_dropout, attention, s_seq_len)
        self.init_state_sym = initial_state_symbol(t_num_lstm_layer, t_num_hidden)

        # initialize states for LSTM
        forward_source_init_c = [('forward_source_l%d_init_c' % l, (batch_size, s_num_hidden)) for l in
                                 range(s_num_lstm_layer)]
        forward_source_init_h = [('forward_source_l%d_init_h' % l, (batch_size, s_num_hidden)) for l in
                                 range(s_num_lstm_layer)]
        backward_source_init_c = [('backward_source_l%d_init_c' % l, (batch_size, s_num_hidden)) for l in
                                  range(s_num_lstm_layer)]
        backward_source_init_h = [('backward_source_l%d_init_h' % l, (batch_size, s_num_hidden)) for l in
                                  range(s_num_lstm_layer)]
        source_init_states = forward_source_init_c + forward_source_init_h + backward_source_init_c + backward_source_init_h

        target_init_c = [('target_l%d_init_c' % l, (batch_size, t_num_hidden)) for l in range(t_num_lstm_layer)]
        target_init_h = [('target_l%d_init_h' % l, (batch_size, t_num_hidden)) for l in range(t_num_lstm_layer)]
        target_init_states = target_init_c + target_init_h

        encode_data_shape = [("source", (batch_size, s_seq_len))]
        decode_data_shape = [("target", (batch_size,))]
        attend_state_shapes = [("attended", (batch_size, s_num_hidden * 2 * s_seq_len))]
        init_state_shapes = [("encoded", (batch_size, s_num_hidden * 2))]

        encode_input_shapes = dict(source_init_states + encode_data_shape)
        decode_input_shapes = dict(target_init_states + decode_data_shape + attend_state_shapes)
        init_input_shapes = dict(init_state_shapes)
        self.encode_executor = self.encode_sym.simple_bind(ctx=ctx, grad_req='null', **encode_input_shapes)
        self.decode_executor = self.decode_sym.simple_bind(ctx=ctx, grad_req='null', **decode_input_shapes)
        self.init_state_executor = self.init_state_sym.simple_bind(ctx=ctx, grad_req='null', **init_input_shapes)

        for key in self.encode_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.encode_executor.arg_dict[key])
        for key in self.decode_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.decode_executor.arg_dict[key])
        for key in self.init_state_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.init_state_executor.arg_dict[key])

        encode_state_name = []
        decode_state_name = []
        for i in range(s_num_lstm_layer):
            encode_state_name.append("forward_source_l%d_init_c" % i)
            encode_state_name.append("forward_source_l%d_init_h" % i)
            encode_state_name.append("backward_source_l%d_init_c" % i)
            encode_state_name.append("backward_source_l%d_init_h" % i)
        for i in range(t_num_lstm_layer):
            decode_state_name.append("target_l%d_init_c" % i)
            decode_state_name.append("target_l%d_init_h" % i)

        self.encode_states_dict = dict(zip(encode_state_name, self.encode_executor.outputs))
        self.decode_states_dict = dict(zip(decode_state_name, self.decode_executor.outputs[1:]))

    def encode(self, input_data):
        for key in self.encode_states_dict.keys():
            self.encode_executor.arg_dict[key][:] = 0.
        input_data.copyto(self.encode_executor.arg_dict["source"])
        self.encode_executor.forward()
        last_encoded = self.encode_executor.outputs[0]
        all_encoded = self.encode_executor.outputs[1]
        return last_encoded, all_encoded

    def decode_forward(self, last_encoded, all_encoded, input_data, new_seq):
        if new_seq:
            last_encoded.copyto(self.init_state_executor.arg_dict["encoded"])
            self.init_state_executor.forward()
            init_hs = self.init_state_executor.outputs[0]
            init_hs.copyto(self.decode_executor.arg_dict["target_l0_init_h"])
            self.decode_executor.arg_dict["target_l0_init_c"][:] = 0.0
            all_encoded.copyto(self.decode_executor.arg_dict["attended"])
        input_data.copyto(self.decode_executor.arg_dict["target"])
        self.decode_executor.forward()

        prob = self.decode_executor.outputs[0].asnumpy()

        self.decode_executor.outputs[1].copyto(self.decode_executor.arg_dict["target_l0_init_c"])
        self.decode_executor.outputs[2].copyto(self.decode_executor.arg_dict["target_l0_init_h"])

        attention_weights = self.decode_executor.outputs[3].asnumpy()

        return prob, attention_weights

    def decode_forward_with_state(self, last_encoded, all_encoded, input_data, state, new_seq):
        if new_seq:
            last_encoded.copyto(self.init_state_executor.arg_dict["encoded"])
            self.init_state_executor.forward()
            init_hs = self.init_state_executor.outputs[0]
            # init_hs.copyto(self.decode_executor.arg_dict["target_l0_init_h"])
            self.decode_executor.arg_dict["target_l0_init_c"][:] = 0.0
            state = LSTMState(c=self.decode_executor.arg_dict["target_l0_init_c"], h=init_hs)
            all_encoded.copyto(self.decode_executor.arg_dict["attended"])
        input_data.copyto(self.decode_executor.arg_dict["target"])
        state.c.copyto(self.decode_executor.arg_dict["target_l0_init_c"])
        state.h.copyto(self.decode_executor.arg_dict["target_l0_init_h"])
        self.decode_executor.forward()

        prob = self.decode_executor.outputs[0]

        c = self.decode_executor.outputs[1]
        h = self.decode_executor.outputs[2]

        attention_weights = self.decode_executor.outputs[3]

        return prob, attention_weights, LSTMState(c=c, h=h)


def bidirectional_encode_symbol(s_num_lstm_layer, s_seq_len, use_masking, s_vocab_size, s_num_hidden, s_num_embed,
                                s_dropout):
    encoder = BiDirectionalLstmEncoder(seq_len=s_seq_len, use_masking=use_masking, state_dim=s_num_hidden,
                                       input_dim=s_vocab_size,
                                       output_dim=0,
                                       vocab_size=s_vocab_size, embed_dim=s_num_embed,
                                       dropout=s_dropout, num_of_layer=s_num_lstm_layer)
    forward_hidden_all, backward_hidden_all, bi_hidden_all = encoder.encode()
    concat_encoded = mx.sym.Concat(*bi_hidden_all, dim=1)
    encoded_for_init_state = mx.sym.Concat(forward_hidden_all[-1], backward_hidden_all[0], dim=1,
                                           name='encoded_for_init_state')
    return mx.sym.Group([encoded_for_init_state, concat_encoded])


def lstm_attention_decode_symbol(t_num_lstm_layer, t_seq_len, t_vocab_size, t_num_hidden, t_num_embed, t_num_label,
                                 t_dropout,
                                 attention, source_seq_len):
    data = mx.sym.Variable("target")
    seqidx = 0

    embed_weight = mx.sym.Variable("target_embed_weight")
    cls_weight = mx.sym.Variable("target_cls_weight")
    cls_bias = mx.sym.Variable("target_cls_bias")

    input_weight = mx.sym.Variable("target_input_weight")
    # input_bias = mx.sym.Variable("target_input_bias")

    param_cells = []
    last_states = []

    for i in range(t_num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("target_l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("target_l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("target_l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("target_l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
                          h=mx.sym.Variable("target_l%d_init_h" % i))
        # state = LSTMState(c=mx.sym.Variable("target_l%d_init_c" % i),
        #                   h=init_hs[i])
        last_states.append(state)
    assert (len(last_states) == t_num_lstm_layer)

    hidden = mx.sym.Embedding(data=data,
                              input_dim=t_vocab_size + 1,
                              output_dim=t_num_embed,
                              weight=embed_weight,
                              name="target_embed")

    all_encoded = mx.sym.Variable("attended")
    encoded = mx.sym.SliceChannel(data=all_encoded, axis=1, num_outputs=source_seq_len)
    weights, weighted_encoded = attention.attend(attended=encoded, concat_attended=all_encoded,
                                                 state=last_states[0].h,
                                                 attend_masks=None,
                                                 use_masking=False)
    con = mx.sym.Concat(hidden, weighted_encoded)
    hidden = mx.sym.FullyConnected(data=con, num_hidden=t_num_embed,
                                   weight=input_weight, no_bias=True, name='input_fc')
    # hidden = mx.sym.Activation(data=hidden, act_type='tanh', name='input_act')

    # stack LSTM
    for i in range(t_num_lstm_layer):
        if i == 0:
            dp = 0.
        else:
            dp = t_dropout
        next_state = lstm(t_num_hidden, indata=hidden,
                          prev_state=last_states[i],
                          param=param_cells[i],
                          seqidx=seqidx, layeridx=i, dropout=dp)
        hidden = next_state.h
        last_states[i] = next_state

    fc = mx.sym.FullyConnected(data=hidden, num_hidden=t_num_label,
                               weight=cls_weight, bias=cls_bias, name='target_pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='target_softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    output.append(weights)
    return mx.sym.Group(output)
