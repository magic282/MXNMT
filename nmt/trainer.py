# -*- coding: utf-8 -*-

import mxnet as mx
import logging

import xconfig
from xsymbol import sym_gen
from xcallback import BatchCheckpoint, CheckBLEUBatch
from xutils import read_content, load_vocab, sentence2id
from xmetric import Perplexity
from masked_bucket_io import MaskedBucketSentenceIter


def get_LSTM_shape():
    # initalize states for LSTM

    forward_source_init_c = [('forward_source_l%d_init_c' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                             range(xconfig.num_lstm_layer)]
    forward_source_init_h = [('forward_source_l%d_init_h' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                             range(xconfig.num_lstm_layer)]
    backward_source_init_c = [('backward_source_l%d_init_c' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                              range(xconfig.num_lstm_layer)]
    backward_source_init_h = [('backward_source_l%d_init_h' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                              range(xconfig.num_lstm_layer)]
    source_init_states = forward_source_init_c + forward_source_init_h + backward_source_init_c + backward_source_init_h

    target_init_c = [('target_l%d_init_c' % l, (xconfig.batch_size, xconfig.num_hidden)) for l in
                     range(xconfig.num_lstm_layer)]
    # target_init_h = [('target_l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    target_init_states = target_init_c
    return source_init_states, target_init_states


def train():
    # load vocabulary
    source_vocab = load_vocab(xconfig.source_vocab_path, xconfig.special_words)
    target_vocab = load_vocab(xconfig.target_vocab_path, xconfig.special_words)

    logging.info('source_vocab size: {0}'.format(len(source_vocab)))
    logging.info('target_vocab size: {0}'.format(len(target_vocab)))

    # get states shapes
    source_init_states, target_init_states = get_LSTM_shape()

    # build data iterator
    data_train = MaskedBucketSentenceIter(xconfig.train_source, xconfig.train_target, source_vocab, target_vocab,
                                          xconfig.buckets, xconfig.batch_size,
                                          source_init_states, target_init_states, seperate_char='\n',
                                          text2id=sentence2id, read_content=read_content,
                                          max_read_sample=xconfig.train_max_samples)

    # data_dev = MaskedBucketSentenceIter(xconfig.dev_source, xconfig.dev_source, source_vocab, target_vocab,
    #                                     xconfig.buckets, xconfig.batch_size,
    #                                     source_init_states, target_init_states, seperate_char='\n',
    #                                     text2id=sentence2id, read_content=read_content,
    #                                     max_read_sample=xconfig.dev_max_samples)

    # Train a LSTM network as simple as feedforward network
    optimizer = mx.optimizer.AdaDelta(clip_gradient=10.0)
    # optimizer = mx.optimizer.Adam(clip_gradient=10.0)
    _arg_params = None

    if xconfig.use_resuming:
        logging.info("Try resuming from {0} {1}".format(xconfig.resume_model_prefix, xconfig.resume_model_number))
        try:
            _, __arg_params, __ = mx.model.load_checkpoint(xconfig.resume_model_prefix, xconfig.resume_model_number)
            logging.info("Resume succeeded.")
            _arg_params = __arg_params
        except:
            logging.error('Resume failed.')

    model = mx.model.FeedForward(ctx=xconfig.train_device,
                                 symbol=sym_gen(len(source_vocab), len(target_vocab)),
                                 num_epoch=xconfig.num_epoch,
                                 optimizer=optimizer,
                                 initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=3),  # 35
                                 arg_params=_arg_params,
                                 )

    # Fit it
    model.fit(X=data_train,
              eval_metric=mx.metric.np(Perplexity),
              # eval_data=data_train,
              batch_end_callback=[mx.callback.Speedometer(xconfig.batch_size, xconfig.show_every_x_batch),
                                  BatchCheckpoint(save_name=xconfig.checkpoint_name,
                                                  per_x_batch=xconfig.checkpoint_freq_batch),
                                  # CheckBLEUBatch(start_epoch=xconfig.eval_start_epoch,
                                  #                per_batch=xconfig.eval_per_x_batch, use_beam=True,
                                  #                beam_size=xconfig.beam_size),
                                  ],
              epoch_end_callback=[mx.callback.do_checkpoint(xconfig.model_save_name, xconfig.model_save_freq),
                                  ])
