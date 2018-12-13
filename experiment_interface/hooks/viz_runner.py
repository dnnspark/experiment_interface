import os
import time
import visdom
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from experiment_interface.hooks import Hook
from experiment_interface.logger import get_train_logger
from experiment_interface.plot_utils import plot_trainval_loss, plot_val_lossacc
from experiment_interface.common import DebugMode


# import matplotlib
# import matplotlib.pyplot as plt

VISDOM_PORT = 8097

RECORD_FILE = 'train_record.csv'

sns.set_style('ticks', {'axes.grid': True})
sns.set_context('talk')


class VisdomRunner(Hook):

    def __init__(self, refresh_interval=10., is_master=False, env='main', **other_kwargs):
        self.refresh_interval = refresh_interval
        self.is_master = is_master
        self.env = env
        self.win = None
        self.init(**other_kwargs)

    def init(self, **kwargs):
        pass

    def set_refresh_interval(self, interval):
        self.refresh_interval = interval

    def _refresh(self, context):
        plt.figure(self.fignumber)
        plt.clf()
        ax = plt.gca()
        self.refresh(context, ax)
        self.win = self.viz.matplot(plt, win=self.win, env=self.env)

    def refresh(self, context, ax):
        raise NotImplementedError('\'refresh\' method must be implemented.')

    def before_loop(self, context):

        if context.debug_mode in (DebugMode.DEBUG, DebugMode.DEV):
            self.refresh_interval = 2.

        logger = get_train_logger()
        if self.is_master:

            # set up visdom.
            cmd = 'tmux kill-session -t visdom_server'
            logger.info(cmd)
            os.system(cmd)
            time.sleep(.1)

            cmd = 'tmux new-session -d -s "visdom_server"'
            logger.info(cmd)
            os.system(cmd)
            time.sleep(.1)

            cmd = 'tmux send-keys -t visdom_server ". activate && python -m visdom.server" Enter'
            logger.info(cmd)
            os.system(cmd)
            # time.sleep(1.)
            logger.info('Wait 10 seconds for Visdom server...')
            time.sleep(3.)

        self.viz = visdom.Visdom(port=VISDOM_PORT, server="http://localhost")
        if not self.viz.check_connection():
            raise RuntimeError('Visdom failed to connect.')
        else:
            logger.info('Visdom client connected.')
        self.last_refreshed = time.time()

        # plt.figure()
        fig = plt.figure()
        self.fignumber = fig.number
        logger.info('VisdomRunner: fignumber=%d' % self.fignumber)

    def after_step(self, context):

        if time.time() - self.last_refreshed > self.refresh_interval:
            logger = get_train_logger()
            logger.info('refreshing visdom runner.')
            self._refresh(context)
            self.last_refreshed = time.time()

    def after_loop(self, context):
        logger = get_train_logger()
        logger.info('refreshing visdom runner.')
        self._refresh(context)
        self.last_refreshed = time.time()


class TrainValLossViz(VisdomRunner):

    def refresh(self, context, ax):

        train_record_file = context.trainer.train_record_file
        df = pd.read_csv(train_record_file, index_col=0)
        if len(df) == 0:
            return None

        plot_trainval_loss(df, ax)


class ValLossAccViz(VisdomRunner):

    def refresh(self, context, ax):

        train_record_file = context.trainer.train_record_file
        df = pd.read_csv(train_record_file, index_col=0)
        if len(df) == 0:
            return None

        plot_val_lossacc(df, ax)

