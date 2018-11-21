import os
import torch
from experiment_interface.logger import get_train_logger

logger = get_train_logger()

class Hook():

    def before_loop(self, context):
        pass

    def before_step(self, context):
        pass

    def after_step(self, context):
        pass

    def after_loop(self, context):
        pass

class StopAtStep(Hook):

    def __init__(self, stop_at):
        self.stop_at = stop_at

    def before_loop(self, context):
        if context.debug:
            self.stop_at = 30

    def after_step(self, context):
        if context.step == self.stop_at:
            context.set_exit()

class SaveNetAtLast(Hook):

    def __init__(self, net_name=None):
        self.net_name = net_name  or 'net'

    def before_loop(self, context):
        self.cache_dir = context.trainer.result_dir

    def after_loop(self, context):
        step = context.step
        if step % 1000 == 0:
            filename = '%s-%03dk.pth' % (self.net_name, step//1000)
        else:
            assert step < 10000
            filename = '%s-%04d.pth' % (self.net_name, step)
        path_to_save = os.path.join(self.cache_dir, filename)
        logger.info('Saving net: %s' % path_to_save)
        torch.save(context.trainer.net.state_dict(), path_to_save)
