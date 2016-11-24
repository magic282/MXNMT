import mxnet as mx


class BasicAttention:
    def __init__(self, batch_size, seq_len, attend_dim, state_dim):
        self.e_weight_W = mx.sym.Variable('energy_W_weight', shape=(state_dim, state_dim))
        self.e_weight_U = mx.sym.Variable('energy_U_weight', shape=(attend_dim, state_dim))
        self.e_weight_v = mx.sym.Variable('energy_v_bias', shape=(state_dim, 1))
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.attend_dim = attend_dim
        self.state_dim = state_dim
        self.pre_compute_buf = {}

    def getHdotU(self, attended, idx):
        if idx not in self.pre_compute_buf:
            h = attended[idx]  # (batch, attend_dim)
            expr = mx.sym.dot(h, self.e_weight_U, name='_energy_1_{0:03d}'.format(idx))
            self.pre_compute_buf[idx] = expr
        return self.pre_compute_buf[idx]

    def attend(self, attended, concat_attended, state, attend_masks, use_masking):
        '''

        :param attended: list [seq_len, (batch, attend_dim)]
        :param concat_attended:  (batch, seq_len, attend_dim )
        :param state: (batch, state_dim)
        :param attend_masks: list [seq_len, (batch, 1)]
        :param use_masking: boolean
        :return:
        '''
        energy_all = []
        pre_compute = mx.sym.dot(state, self.e_weight_W, name='_energy_0')
        for idx in range(self.seq_len):
            h = attended[idx]  # (batch, attend_dim)
            energy = pre_compute + mx.sym.dot(h, self.e_weight_U,
                                              name='_energy_1_{0:03d}'.format(idx))  # (batch, state_dim)
            # energy = pre_compute + self.getHdotU(attended, idx)
            energy = mx.sym.Activation(energy, act_type="tanh",
                                       name='_energy_2_{0:03d}'.format(idx))  # (batch, state_dim)
            energy = mx.sym.dot(energy, self.e_weight_v, name='_energy_3_{0:03d}'.format(idx))  # (batch, 1)
            if use_masking:
                energy = energy * attend_masks[idx] + (1.0 - attend_masks[idx]) * (-10000.0)  # (batch, 1)
            energy_all.append(energy)

        all_energy = mx.sym.Concat(*energy_all, dim=1, name='_all_energy_1')  # (batch, seq_len)

        alpha = mx.sym.SoftmaxActivation(all_energy, name='_alpha_1')  # (batch, seq_len)
        alpha = mx.sym.Reshape(data=alpha, shape=(self.batch_size, self.seq_len, 1),
                               name='_alpha_2')  # (batch, seq_len, 1)

        weighted_attended = mx.sym.broadcast_mul(alpha, concat_attended,
                                                 name='_weighted_attended_1')  # (batch, seq_len, attend_dim)
        weighted_attended = mx.sym.sum(data=weighted_attended, axis=1,
                                       name='_weighted_attended_2')  # (batch,  attend_dim)
        return alpha, weighted_attended
