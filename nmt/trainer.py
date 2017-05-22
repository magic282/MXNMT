# -*- coding: utf-8 -*-

import mxnet as mx
import logging

import xconfig
from xsymbol import sym_gen
from xcallback import BatchCheckpoint, CheckBLEUBatch
from xutils import read_content, load_vocab, sentence2id
from xmetric import Perplexity, MyMakeLoss
# from masked_bucket_io import MaskedBucketSentenceIter
from masked_bucket_io_new import MaskedBucketSentenceIter


def get_GRU_shape():
    # initalize states for LSTM

    forward_source_init_h = [('forward_source_l%d_init_h' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                             range(xconfig.num_lstm_layer)]
    backward_source_init_h = [('backward_source_l%d_init_h' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                              range(xconfig.num_lstm_layer)]
    source_init_states = forward_source_init_h + backward_source_init_h

    target_init_c = [('target_l%d_init_c' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                     range(xconfig.num_lstm_layer)]
    # target_init_h = [('target_l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    target_init_states = []
    return source_init_states, target_init_states


def train():
    # load vocabulary
    source_vocab = load_vocab(xconfig.source_vocab_path, xconfig.special_words)
    target_vocab = load_vocab(xconfig.target_vocab_path, xconfig.special_words)

    logging.info('source_vocab size: {0}'.format(len(source_vocab)))
    logging.info('target_vocab size: {0}'.format(len(target_vocab)))

    # get states shapes
    source_init_states, target_init_states = get_GRU_shape()
    # source_init_states, target_init_states = get_LSTM_shape()

    # build data iterator
    data_train = MaskedBucketSentenceIter(xconfig.train_source, xconfig.train_target, source_vocab, target_vocab,
                                          xconfig.buckets, xconfig.batch_size,
                                          source_init_states, target_init_states,
                                          text2id=sentence2id, read_content=read_content,
                                          max_read_sample=xconfig.train_max_samples)

    # data_dev = MaskedBucketSentenceIter(xconfig.dev_source, xconfig.dev_source, source_vocab, target_vocab,
    #                                     xconfig.buckets, xconfig.batch_size,
    #                                     source_init_states, target_init_states, seperate_char='\n',
    #                                     text2id=sentence2id, read_content=read_content,
    #                                     max_read_sample=xconfig.dev_max_samples)

    # Train a LSTM network as simple as feedforward network
    # optimizer = mx.optimizer.AdaDelta(clip_gradient=10.0)
    optimizer = mx.optimizer.Adam(clip_gradient=10.0, rescale_grad=1.0 / xconfig.batch_size)
    # optimizer = mx.optimizer.SGD(clip_gradient=10, learning_rate=0.01, rescale_grad=1.0 / xconfig.batch_size)
    _arg_params = None

    if xconfig.use_resuming:
        logging.info("Try resuming from {0} {1}".format(xconfig.resume_model_prefix, xconfig.resume_model_number))
        try:
            _, __arg_params, __ = mx.model.load_checkpoint(xconfig.resume_model_prefix, xconfig.resume_model_number)
            logging.info("Resume succeeded.")
            _arg_params = __arg_params
        except:
            logging.error('Resume failed.')

    model = mx.mod.BucketingModule(
        sym_gen=sym_gen(len(source_vocab) + 1, len(target_vocab) + 1),
        default_bucket_key=data_train.default_bucket_key,
        context=xconfig.train_device,
    )

    # Fit it
    model.fit(train_data=data_train,
              # eval_metric=mx.metric.np(Perplexity),
              eval_metric=mx.metric.CustomMetric(Perplexity),
              # eval_metric=mx.metric.np(MyMakeLoss),
              batch_end_callback=[mx.callback.Speedometer(xconfig.batch_size, xconfig.show_every_x_batch), ],
              # optimizer='sgd',
              # optimizer_params={'clip_gradient': 10.0, },
              initializer=mx.init.Xavier(factor_type="in", magnitude=2.34, rnd_type='gaussian'),
              optimizer=optimizer,
              num_epoch=10,
              )
