import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

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

def plot_trainval_loss(df, ax):
    '''
    Plot (smoothed) train loss and val loss.

    Assumptions:
        - 'batch_loss' and 'val_loss' type for train/val losses.
    '''

    step = df['step'].max()

    raw_batch_loss = df.loc[ df['type'] == 'batch_loss', ['step', 'value'] ]
    val_loss = df.loc[ df['type'] == 'val_loss', ['step', 'value'] ]

    win_size = 20 if step > 100 else 1
    smoothed = raw_batch_loss['value'].rolling(win_size, center=True, win_type='parzen').mean()
    smoothed_batch_loss = pd.DataFrame({'step': raw_batch_loss['step'], 'value': smoothed})

    smoothed_batch_loss['type'] = 'train'
    val_loss['type'] = 'val'

    ymax = raw_batch_loss['value'].quantile(.98)

    # ax = plt.gca()
    ax = raw_batch_loss.plot(x='step', y='value', color=BRIGHTER_COLORS5[0], legend=False, ax=ax)

    if step > 1000:
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(format_tick))

    df = pd.concat([smoothed_batch_loss, val_loss], ignore_index=True, sort=False)
    num_types = len(df['type'].unique())
    sns.lineplot(data=df, x='step', y='value', hue='type', palette=COLORS5[:num_types], ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[2:], labels=labels[2:])

    ax.set_ylabel('loss')
    ax.set_ylim([0., ymax])

    plt.tight_layout()

def plot_val_lossacc(df, ax):
    '''
    Plot loss and test metric on validation set.

    Assumptions:
        - 'val_loss' and 'val_acc' type for val loss/accuracy.
    '''

    val_loss = df.loc[ df['type'] == 'val_loss', ['step', 'value'] ]
    val_acc = df.loc[ df['type'] == 'val_acc', ['step', 'value'] ]

    if len(val_loss) == 0 or len(val_acc) == 0:
        return None

    step = df['step'].max()

    val_loss['type'] = 'val loss'
    val_acc['type'] = 'val acc'

    # ax = plt.gca()
    ax = val_loss.plot(
            x = 'step', y='value', color=COLORS5[1], legend=False, ax=ax)

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
