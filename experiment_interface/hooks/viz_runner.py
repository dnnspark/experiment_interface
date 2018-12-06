import os
import time
import visdom
from experiment_interface.hooks import Hook
from experiment_interface.logger import get_train_logger

import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

VISDOM_PORT = 8097

RECORD_FILE = 'train_record.csv'

sns.set_style('ticks', {'axes.grid': True})
sns.set_context('talk')

COLORS5 = [
    # these are second-last elements of single-hue colors from http://colorbrewer2.org
    '#3182bd', # blue
    '#de2d26', # red
    '#31a354', # green
    '#756bb1', # purple 
    '#636363', # grey

]

BRIGHTER_COLORS5 = [
    # these are third elements of single-hue colors from http://colorbrewer2.org
    '#9ecae1', # blue
    '#fc9272', # red
    '#a1d99b', # green
    '#bcbddc', # purple
    '#bdbdbd', # grey
]


def format_tick(x, p):
    y = x / 1000.
    return '%.1fk' % y

def plot_trainval_loss(scalar_record_file, context):
    '''
    Plot (smoothed) train loss and val loss.

    Assumptions:
        - 'batch_loss' and 'val_loss' type for train/val losses.
    '''
    df = pd.read_csv(scalar_record_file, index_col=0)
    if len(df) == 0:
        return None

    step = df['step'].max()

    raw_batch_loss = df.loc[ df['type'] == 'batch_loss', ['step', 'value'] ]
    val_loss = df.loc[ df['type'] == 'val_loss', ['step', 'value'] ]

    win_size = 20 if step > 100 else 1
    smoothed = raw_batch_loss['value'].rolling(win_size, center=True, win_type='parzen').mean()
    smoothed_batch_loss = pd.DataFrame({'step': raw_batch_loss['step'], 'value': smoothed})

    smoothed_batch_loss['type'] = 'train'
    val_loss['type'] = 'val'

    # ymax = raw_batch_loss['value'].mean() * 2.
    ymax = raw_batch_loss['value'].quantile(.95)

    ax = raw_batch_loss.plot(x='step', y='value', color=BRIGHTER_COLORS5[0], legend=False)

    if step > 1000:
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(format_tick))

    df = pd.concat([smoothed_batch_loss, val_loss], ignore_index=True)
    num_types = len(df['type'].unique())
    sns.lineplot(data=df, x='step', y='value', hue='type', palette=COLORS5[:num_types], ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[2:], labels=labels[2:])

    ax.set_ylabel('loss')
    ax.set_ylim([0., ymax])

    plt.tight_layout()

def plot_val_lossacc(scalar_record_file, context):
    '''
    Plot loss and test metric on validation set.

    Assumptions:
        - 'val_loss' and 'val_acc' type for val loss/accuracy.
    '''
    df = pd.read_csv(scalar_record_file, index_col=0)
    if len(df) == 0:
        return None

    val_loss = df.loc[ df['type'] == 'val_loss', ['step', 'value'] ]
    val_acc = df.loc[ df['type'] == 'val_acc', ['step', 'value'] ]

    if len(val_loss) == 0 or len(val_acc) == 0:
        return None

    step = df['step'].max()

    val_loss['type'] = 'val loss'
    val_acc['type'] = 'val acc'

    ax = val_loss.plot(
            x = 'step', y='value', color=COLORS5[1], legend=False)

    ymax = val_loss['value'].quantile(.95)
    ax.set_ylim([0., ymax])
    ax.set_ylabel('loss')

    if step > 1000:
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(format_tick))

    ax2 = ax.twinx()
    val_acc.plot(
            x = 'step', y='value', ax=ax2, color=COLORS5[2], legend=False)
    ax2.set_ylim([0., 1.])
    ax2.set_ylabel('accuracy')

    lgnd = ax.figure.legend()
    lgnd.texts[0].set_text('val loss')
    lgnd.texts[1].set_text('val acc')

    plt.tight_layout()

class VisdomRunner(Hook):

    def __init__(self, plot_fn, refresh_interval=10., is_master=True):
        self.plot_fn = plot_fn
        self.refresh_interval = refresh_interval
        self.is_master = is_master
        # self.viz = None

    def refresh(self, context):

        self.plot_fn(context.trainer.scalar_record_file, context)
        self.win = self.viz.matplot(plt, win=self.win, env=self.env)

        # plot_loss(context.trainer.scalar_record_file, context)
        # self.win = self.viz.matplot(plt, win=self.win, env=self.env)

        # plot_val(context.trainer.scalar_record_file, context)
        # self.win2 = self.viz.matplot(plt, win=self.win2, env=self.env)

    def before_loop(self, context):

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
            logger.info('Wait 3 seconds for Visdom server...')
            time.sleep(3.)

        self.viz = visdom.Visdom(port=VISDOM_PORT, server="http://localhost")
        if not self.viz.check_connection():
            raise RuntimeError('Visdom failed to connect.')
        else:
            logger.info('Visdom client connected.')
        self.last_refreshed = time.time()

        self.win = None
        self.win2 = None
        self.env = 'main'

    def after_step(self, context):

        if time.time() - self.last_refreshed > self.refresh_interval:
            logger = get_train_logger()
            logger.info('refreshing visdom runner.')
            self.refresh(context)
            self.last_refreshed = time.time()

    def after_loop(self, context):
        self.refresh(context)
