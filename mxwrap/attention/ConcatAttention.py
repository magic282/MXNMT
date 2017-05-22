import mxnet as mx


class ConcatAttention:
    def __init__(self, batch_size, attend_dim, state_dim):
        self.e_weight_W = mx.sym.Variable('energy_W_weight', shape=(state_dim, state_dim))
        self.e_weight_U = mx.sym.Variable('energy_U_weight', shape=(attend_dim, state_dim))
        self.e_weight_v = mx.sym.Variable('energy_v_bias', shape=(state_dim, 1))
        self.batch_size = batch_size
        self.attend_dim = attend_dim
        self.state_dim = state_dim

    def pre_compute(self, attended):
        seq_len = len(attended)
        res = [None for i in range(seq_len)]
        for idx in range(seq_len):
            h = attended[idx]
            res[idx] = mx.sym.dot(h, self.e_weight_U, name='_energy_1_{0:03d}'.format(idx))
        return res

    def attend(self, source_pre_computed, attended, concat_attended, state, attend_masks, use_masking):
        '''

        :param attended: list [seq_len, (batch, attend_dim)]
        :param concat_attended:  (batch, seq_len, attend_dim )
        :param state: (batch, state_dim)
        :param attend_masks: list [seq_len, (batch, 1)]
        :param use_masking: boolean
        :return:
        '''
        seq_len = len(attended)
        energy_all = []
        pre_compute = mx.sym.dot(state, self.e_weight_W, name='_energy_0')
        for idx in range(seq_len):
            energy = pre_compute + source_pre_computed[idx]
            energy = mx.sym.Activation(energy, act_type="tanh",
                                       name='_energy_2_{0:03d}'.format(idx))  # (batch, state_dim)
            energy = mx.sym.dot(energy, self.e_weight_v, name='_energy_3_{0:03d}'.format(idx))  # (batch, 1)
            if use_masking:
                energy = energy * attend_masks[idx] + (1.0 - attend_masks[idx]) * (-10000.0)  # (batch, 1)
            energy_all.append(energy)

        all_energy = mx.sym.Concat(*energy_all, dim=1, name='_all_energy_1')  # (batch, seq_len)

        alpha = mx.sym.SoftmaxActivation(all_energy, name='_alpha_1')  # (batch, seq_len)
        alpha = mx.sym.Reshape(data=alpha, shape=(self.batch_size, seq_len, 1),
                               name='_alpha_2')  # (batch, seq_len, 1)

        weighted_attended = mx.sym.broadcast_mul(alpha, concat_attended,
                                                 name='_weighted_attended_1')  # (batch, seq_len, attend_dim)
        weighted_attended = mx.sym.sum(data=weighted_attended, axis=1,
                                       name='_weighted_attended_2')  # (batch,  attend_dim)
        return alpha, weighted_attended

    def pre_compute_fast(self, attended):
        seq_len = len(attended)
        buf = []
        for s in attended:
            buf.append(mx.sym.expand_dims(data=s, axis=0))
        time_major_concat = mx.sym.concat(*buf, dim=0, name='time_major_concat')  # (seq, batch, dim)
        time_major_concat = mx.sym.dot(time_major_concat, self.e_weight_U, name='_expr01')  # (seq, batch, dim)
        return time_major_concat

    def attend_fast(self, source_pre_computed, seq_len, state, attend_masks, use_masking):
        '''

        :param source_pre_computed:
        :param seq_len:
        :param state:
        :param attend_masks:
        :param use_masking:
        :return:
        '''
        energy_all = []
        pre_compute = mx.sym.dot(state, self.e_weight_W, name='_energy_00')  # (batch, dim)
        pre_compute = mx.sym.expand_dims(data=pre_compute, axis=0, name='_energy_10')  # (1, batch, dim)

        energy = mx.sym.broadcast_add(source_pre_computed, pre_compute, name='_b_add')
        energy = mx.sym.Activation(energy, act_type="tanh", name='_energy_20')  # (seq, batch, dim)
        energy = mx.sym.dot(energy, self.e_weight_v, name='_energy_30')  # (seq, batch, 1)
        energy = mx.sym.reshape(energy, shape=(seq_len, -1))  # (seq, batch)
        energy = mx.sym.split(energy, axis=0, num_outputs=seq_len, squeeze_axis=True)  # [seq, (batch,)]

        for idx in range(seq_len):
            this_e = energy[idx]
            if use_masking:
                this_e = this_e * attend_masks[idx] + (1.0 - attend_masks[idx]) * (-1e6)  # (batch,)
                this_e = mx.sym.expand_dims(data=this_e, axis=0, name='_this_e_10')
            energy_all.append(this_e)

        all_energy = mx.sym.Concat(*energy_all, dim=0, name='_all_energy_1')  # (seq, batch)

        alpha = mx.sym.SoftmaxActivation(all_energy, name='_alpha_1')  # (seq, batch)
        alpha = mx.sym.expand_dims(data=alpha, axis=2, name='_alpha_2')  # (seq, batch, 1)

        weighted_attended = mx.sym.broadcast_mul(source_pre_computed, alpha,
                                                 name='_weighted_attended_1')  # (seq, batch, attend_dim)
        weighted_attended = mx.sym.sum(data=weighted_attended, axis=0,
                                       name='_weighted_attended_2')  # (batch, attend_dim)
        return alpha, weighted_attended
