import os
import pandas as pd
import collections
import time
from experiment_interface.hooks import Hook
from experiment_interface.logger import get_train_logger

columns = ['step', 'type', 'value']
Row = collections.namedtuple('Row', ' '.join(columns))

class ScalarRecorder(Hook):
    '''
    Log real-valued number during training.
    '''

    def __init__(self, log_file, flush_interval=100):
        assert log_file.endswith('.csv')
        self.log_file = log_file
        self.flush_interval = flush_interval

    def append(self, row):
        self.dfs.append( pd.DataFrame([ row ], columns=columns) )

    def before_loop(self, context):

        if context.debug:
            self.flush_interval = 10

        log_file = self.log_file

        if os.path.exists(log_file):
            logger = get_train_logger()
            logger.warning('Deleting %s.' % log_file)
            os.remove(log_file)

        df = pd.DataFrame(columns = ['step', 'type', 'value'])
        with open(log_file, 'w') as f:
            df.to_csv(f, header=True)

        self.first_index = 0
        self.dfs = []

        # # set up visdom.
        # cmd = 'tmux kill-session -t visdom_server'
        # logger.info(cmd)
        # os.system(cmd)
        # time.sleep(.1)

        # cmd = 'tmux new-session -d -s "visdom_server"'
        # logger.info(cmd)
        # os.system(cmd)
        # time.sleep(.1)

        # cmd = 'tmux send-keys -t visdom_server "python -m visdom.server" Enter'
        # logger.info(cmd)
        # os.system(cmd)
        # time.sleep(.1)

    def after_step(self, context):

        step = context.step
        if step % self.flush_interval == 0:
            logger = get_train_logger()
            logger.info('ScalarLogger flushing data... step=%d' % step)
            df = pd.concat(self.dfs, ignore_index=True)
            df.index = df.index + self.first_index 
            with open(self.log_file, 'a') as f:
                df.to_csv(f, header=False)
            self.first_index += len(df)
            self.dfs = []

