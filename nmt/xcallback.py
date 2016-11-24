import xconfig

import mxnet as mx
import logging


class BatchCheckpoint(object):
    def __init__(self, save_name, per_x_batch):
        self.save_name = save_name
        self.per_x_batch = per_x_batch
        from mxnet.model import save_checkpoint
        self._save = save_checkpoint

    def __call__(self, params):
        # batch_end_params = BatchEndParam(epoch=epoch,
        #                                  nbatch=nbatch,
        #                                  eval_metric=eval_metric,
        #                                  locals=locals())

        if params.nbatch % self.per_x_batch == 0:
            executor_manager = params.locals['executor_manager']
            param_names = executor_manager.param_names
            param_arrays = executor_manager.param_arrays

            param_dict = {}
            for idx, name in enumerate(param_names):
                param_dict[name] = param_arrays[idx][0]

            self._save(self.save_name, 0, params.locals['symbol'],
                       param_dict, params.locals['aux_params'])
            # TODO is this the correct way to save aux_params ?


class CheckBLEUBatch(object):
    def __init__(self, start_epoch, per_batch, use_beam=False, beam_size=-1):
        self.best_bleu = -1.0
        self.best_epoch = -1
        self.start_epoch = start_epoch
        self.per_batch = per_batch
        self.use_beam_search = use_beam
        self.beam_size = beam_size
        from mxnet.model import save_checkpoint
        self._save = save_checkpoint
        # TODO ugly code 2333
        from tester import test_use_model_param
        self.bleu_computer = test_use_model_param

    def __call__(self, params):
        # batch_end_params = BatchEndParam(epoch=epoch,
        #                                  nbatch=nbatch,
        #                                  eval_metric=eval_metric,
        #                                  locals=locals())

        if params.nbatch % self.per_batch == 0:
            if params.epoch < self.start_epoch:
                print('Too early to check BLEU at epoch {0}'.format(params.epoch))
                return
            logging.info('Checking BLEU for epoch {0} batch {1}'.format(params.epoch, params.nbatch))
            gold = xconfig.dev_source
            test = xconfig.dev_source
            output = xconfig.dev_output

            executor_manager = params.locals['executor_manager']
            param_names = executor_manager.param_names
            param_arrays = executor_manager.param_arrays

            param_dict = {}
            for idx, name in enumerate(param_names):
                param_dict[name] = param_arrays[idx][0]

            cur_rouge = self.bleu_computer(arg_params=param_dict, test_file=test, output_file=output, gold_file=gold,
                                           use_beam=self.use_beam_search, beam_size=self.beam_size)
            logging.info('BLEU: {0} @ epoch {1} batch {2}'.format(cur_rouge, params.epoch, params.nbatch))

            if cur_rouge > self.best_bleu:
                logging.info(
                    'Current BLEU: {0} > prev best {1} in epoch {2}'.format(cur_rouge, self.best_bleu,
                                                                            self.best_epoch))
                self.best_bleu = cur_rouge
                self.best_epoch = params.epoch
                logging.info('Saving...')
                self._save("best_bleu", params.epoch + 1, params.locals['symbol'],
                           param_dict, params.locals['aux_params'])
                # TODO is this the correct way to save aux_params ?
