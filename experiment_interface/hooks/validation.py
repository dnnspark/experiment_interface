import torch
import numpy as np
import os
from experiment_interface.hooks import Hook
from experiment_interface.hooks.scalar_recorder import Row
from experiment_interface.evaluator import Evaluator
from experiment_interface.evaluator.metrics import LossMetric
from experiment_interface.logger import get_train_logger
from experiment_interface.common import DebugMode

def _identity(*args):
    return args if len(args) > 1 else args[0]

class ValidationHook(Hook):

    def __init__(self, dataset, interval, name, predict_fn=None, metric=None, save_best=True, cache_dir=None):
        self.dataset = dataset
        self.interval = interval
        self.name = name

        if predict_fn is None:
            predict_fn = _identity
        self.predict_fn = predict_fn

        self.metric = metric

        self.save_best = save_best
        self.cache_dir = cache_dir

    def set_cache_dir(self, cache_dir):
        if self.cache_dir is not None:
            raise ValueError('\'cache_dir\' is not None, and overwriting is not allowed.')

        self.cache_dir = cache_dir

    def before_loop(self, context):

        logger = get_train_logger()

        if context.debug_mode == DebugMode.DEBUG:
            self.interval = 10
        elif context.debug_mode == DebugMode.DEV:
            self.interval = 50

        if self.metric is None:
            self.metric = LossMetric(context.trainer.loss_module_class)
        self.larger_is_better = larger_is_better = self.metric.larger_is_better

        self.batch_size = context.trainer.batch_size
        self.num_workers = context.trainer.num_workers

        self.best_metric = -np.inf if larger_is_better else np.inf
        self.last_saved = None

        if self.cache_dir is not None:
            self.cache_dir = cache_dir = os.path.join(self.cache_dir, self.name)
            logger.info('Creating %s.' % cache_dir)
            os.makedirs(cache_dir)


    def after_step(self, context):

        if context.step % self.interval == 0:
            self._validate(context)

    def after_loop(self, context):
        self._validate(context)

    def _validate(self, context):

        logger = get_train_logger()

        evaluator = Evaluator(
            net = context.trainer.net,
            test_dataset = self.dataset, 
            batch_size = self.batch_size,
            predict_fn = self.predict_fn,
            metric = self.metric,
            num_workers = self.num_workers,
            is_validating = True,
            )

        if self.cache_dir is not None:
            record_file = os.path.join(self.cache_dir, 'step%07d.csv' % context.step)
            evaluator.set_record_file(record_file)

        metric_eval = evaluator.run()
        try:
            metric_eval = metric_eval.detach().cpu().numpy()
        except AttributeError:
            # metric_eval may be not torch.Tensor                
            pass

        logger.info('step=%d | VAL | %s=%.4f' % (context.step, self.name, metric_eval) )
        train_recorder = context.trainer.train_recorder
        if train_recorder is not None:
            train_recorder.append( Row(context.step, self.name, metric_eval) )

        if self.save_best and ( self.larger_is_better == (metric_eval > self.best_metric) ):
            # save net
            step = context.step
            try:
                net_name = context.trainer.net.module._name
            except AttributeError:
                net_name = 'net'
            filename = '%s-%07d.pth' % (net_name, step)

            cache_dir = context.trainer.result_dir

            path_to_save = os.path.join(cache_dir, filename)
            logger.info('Saving net: %s' % path_to_save)
            torch.save(context.trainer.net.state_dict(), path_to_save)

            if self.last_saved is not None:
                logger.info('Deleting %s' % self.last_saved)
                os.remove(self.last_saved)
            self.last_saved = path_to_save

            self.best_metric = metric_eval

        context.add_item({self.name: metric_eval})
        # TODO: this might be a problem when using "test" mode in train.
        # e.g. batchnorm with batch_size=1
        context.trainer.net.train() 


