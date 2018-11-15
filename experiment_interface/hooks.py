import os
import torch
from experiment_interface import Hook

class StopAtStep(Hook):

    def __init__(self, stop_at):
        self.stop_at = stop_at

    def after_step(self, context):
        if context.step == self.stop_at:
            context.set_exit()

class SaveNetAtLast(Hook):

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def after_loop(self, context):
        step = context.step
        if step % 1000 == 0:
            filename = 'net-%03dk.pth' % (step//1000)
        else:
            assert step < 10000
            filename = 'net-%04d.pth' % step
        path_to_save = os.path.join(self.cache_dir, filename)
        torch.save(context.net.state_dict(), path_to_save)
