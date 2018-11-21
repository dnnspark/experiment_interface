import torch
from experiment_interface.hooks import Hook
from experiment_interface.evaluator import Evaluator
from experiment_interface.evaluator.metrics import LossMetric
from experiment_interface.logger import get_train_logger

def _identity(*args):
    return args if len(args) > 1 else args[0]

class ValidationHook(Hook):

    def __init__(self, dataset, interval, name='val', predict_fn=None, metric=None):
        self.dataset = dataset
        self.interval = interval
        self.name = name

        if predict_fn is None:
            predict_fn = _identity
        self.predict_fn = predict_fn

        self.metric = metric

    def before_loop(self, context):

        if context.debug:
            self.interval = 5

        if self.metric is None:
            self.metric = LossMetric(context.trainer.loss_fn)

        self.batch_size = context.trainer.batch_size
        self.num_workers = context.trainer.num_workers

    def after_step(self, context):
        if context.step % self.interval == 0:
            # TODO: set up an evaluator and run.

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
                score = score.detach()
            except:
                pass
            context.add_item({self.name: score})
            context.trainer.net.train()

            logger = get_train_logger()
            logger.info('step=%d | VAL | %s=%.4f' % (context.step, self.name, score) )


