import mxnet as mx
from collections import namedtuple
from .BaseCell import BaseCell


class GRU(BaseCell):
    def __init__(self, name, num_hidden, **kwargs):
        super(BaseCell, self).__init__()
        self.name = name + '_GRU'
        self.num_hidden = num_hidden
        self.W = mx.sym.Variable("{0}_W_weight".format(self.name))
        self.B = mx.sym.Variable("{0}_W_bias".format(self.name))
        self.U = mx.sym.Variable("{0}_U_weight".format(self.name))

    def apply(self, indata, prev_h, seqidx, mask=None):
        xW = mx.sym.FullyConnected(data=indata,
                                   weight=self.W,
                                   bias=self.B,
                                   num_hidden=self.num_hidden * 3,
                                   name="{0}_xW_{1}".format(self.name, seqidx)
                                   )
        # hU = mx.sym.dot(prev_state.h, param.gru_U_weight)
        hU = mx.sym.FullyConnected(data=prev_h,
                                   weight=self.U,
                                   num_hidden=self.num_hidden * 3,
                                   no_bias=True,
                                   name="{0}_hU_{1}".format(self.name, seqidx)
                                   )
        xW_s = mx.sym.split(num_outputs=3, data=xW)
        hU_s = mx.sym.split(num_outputs=3, data=hU)
        r = mx.sym.Activation(data=(xW_s[0] + hU_s[0]), act_type='sigmoid')
        z = mx.sym.Activation(data=(xW_s[1] + hU_s[1]), act_type='sigmoid')
        h1 = mx.sym.Activation(data=(xW_s[2] + r * hU_s[2]), act_type='tanh')

        h = (h1 - prev_h) * z + prev_h
        if mask:
            h = mx.sym.broadcast_mul(mask, h, name='bm_1') + mx.sym.broadcast_mul((1 - mask), prev_h, name='bm_2')
        return h
