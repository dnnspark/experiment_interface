from experiment_interface import Hook

class StopAtStep(Hook):

    def __init__(self, stop_at):
        self.stop_at = stop_at

    def after_step(self, context):
        if context.step == self.stop_at:
            context.set_exit()