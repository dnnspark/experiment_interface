import torch
import numpy as np
import os
from experiment_interface.hooks import Hook
from experiment_interface.hooks.scalar_recorder import Row
from experiment_interface.evaluator import Evaluator
from experiment_interface.evaluator.metrics import LossMetric
from experiment_interface.logger import get_train_logger

def _identity(*args):
    return args if len(args) > 1 else args[0]

class ValidationHook(Hook):

    def __init__(self, dataset, interval, name, predict_fn=None, metric=None, save_best=True):
        self.dataset = dataset
        self.interval = interval
        self.name = name

        if predict_fn is None:
            predict_fn = _identity
        self.predict_fn = predict_fn

        self.metric = metric

        self.save_best = save_best

    def before_loop(self, context):

        if context.debug:
            self.interval = 5

        if self.metric is None:
            self.metric = LossMetric(context.trainer.loss_fn)
        self.larger_is_better = larger_is_better = self.metric.larger_is_better

        self.batch_size = context.trainer.batch_size
        self.num_workers = context.trainer.num_workers

        self.best_metric = -np.inf if larger_is_better else np.inf
        self.last_saved = None

    def after_step(self, context):

        if context.step % self.interval == 0:

            metric = self.metric
            if metric is None:
                metric = LossMetric(context.trainer.loss_fn)

            evaluator = Evaluator(
                net = context.trainer.net,
                test_dataset = self.dataset, 
                batch_size = self.batch_size,
                predict_fn = self.predict_fn,
                metric = self.metric,
                num_workers = self.num_workers,
                is_validating = True,
                )

            score = evaluator.run()
            try:
                score = score.detach().cpu().numpy()
            except AttributeError:
                # score may be not torch.Tensor                
                pass

            logger = get_train_logger()
            logger.info('step=%d | VAL | %s=%.4f' % (context.step, self.name, score) )
            scalar_recorder = context.trainer.scalar_recorder
            if scalar_recorder is not None:
                scalar_recorder.append( Row(context.step, self.name, score) )

            if self.save_best and ( self.larger_is_better == (score > self.best_metric) ):
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

                self.best_metric = score

            context.add_item({self.name: score})
            # TODO: this might be a problem when using "test" mode in train.
            # e.g. batchnorm with batch_size=1
            context.trainer.net.train() 
