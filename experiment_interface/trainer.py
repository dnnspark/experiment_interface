'''
Hooks
- Stop training at step N.
- initialize with pre-trained parameters. 
- freeze parameters until N step.
- validate every N step.
- save net every N step.

hooks schedules:
    - before_loop
    - before_step
    - after_step
    - after_loop

Trainer
- using gpu
- using multiple gpu's
- using hooks

'''

import torch
import logging
import os

# logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S')

# logging.addLevelName(logging.INFO, '')

formatter = logging.Formatter(
    fmt='%(levelname)s %(asctime)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

class TrainContext():

    def __init__(self, net, trainer):
        '''
        A hook can set 'exit_loop' to True to terimnate the training loop.
        '''
        self.net = net
        self.trainer = trainer

        self.step = 0
        self.exit_loop = False
        self.context = {}

    def inc_step(self):
        self.step += 1

    def set_exit(self):
        self.exit_loop = True

    def add_item(self, named_item):
        '''
        named_item: dict
            {name: item}
        '''
        self.context.update(named_item)


class Trainer():

    def __init__(self,
        net,
        train_dataset,
        batch_size,
        loss_fn,
        optimizer,
        result_dir,
        num_workers = None,
        hooks = [],
        log_file = None,
        log_interval = 1,
        ):

        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.net = net 
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.result_dir = result_dir
        self.num_workers = num_workers
        self.hooks = hooks
        self.logger = self.get_logger(os.path.join(self.result_dir, log_file))
        self.log_interval = log_interval

    @classmethod
    def get_logger(cls, log_file=None):
        logger = logging.getLogger('train_logger')

        if len(logger.handlers) == 0:
            # not set up yet.
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        if log_file is not None:
            fh = logging.FileHandler(filename=log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger


    def run(self):

        logger = self.logger

        use_cuda = torch.cuda.is_available()

        if use_cuda and self.num_workers is None:
            raise ValueError('\'num_workers\' must be int, if cuda is available.')

        if not use_cuda and self.num_workers is not None:
            logger.warning('\'num_workers=%d\' is ignored and set to zero, because use_cuda is False.' % self.num_workers)
            self.num_workers = 0

        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, 
            num_workers=self.num_workers, pin_memory=use_cuda)

        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        net = self.net.to(device)
        net = torch.nn.DataParallel(net)
        net.train()

        context = TrainContext(net, self)

        for hook in self.hooks:
            hook.before_loop(context)

        # training loop
        while not context.exit_loop:

            for batch in train_data_loader:
                # Assumption: 
                # - the first element of batch is image batch, the rest are labels
                # - image batch is the only input to the net.
                # - the remaining tensors are fed into loss function as positional args, following the prediction tensors.
                assert isinstance(batch, list) or isinstance(batch, tuple)
                images, labels = batch[0], batch[1:]

                context.inc_step()

                for hook in self.hooks:
                    hook.before_step(context)

                images = images.to(device)
                labels = [label.to(device) for label in labels]
                predictions = net(images)
                if not (isinstance(predictions, list) or isinstance(predictions, tuple)):
                    predictions = [predictions]
                losses = self.loss_fn(*predictions, *labels)
                # Assumption:
                # - self.loss_fn() returns either a 0-dim (scalar) Tensor, or a list/tuple of the following form
                # - [total_loss, (loss_name_1, loss1), (loss_name_2, loss2), ...]
                if isinstance(losses, list) or isinstance(losses, tuple):
                    total_loss, other_losses = losses[0], losses[1:]
                else:
                    assert losses.dim() == 0 
                    total_loss = losses
                    other_losses = []

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if context.step % self.log_interval == 0:
                    logger.info('step=%d | loss=%.4f' % (context.step, total_loss.detach()))

                for hook in self.hooks:
                    hook.after_step(context)

                if context.exit_loop:
                    break;

        for hook in self.hooks:
            hook.after_loop(context)


class Hook():

    def before_loop(self, context):
        pass

    def before_step(self, context):
        pass

    def after_step(self, context):
        pass

    def after_loop(self, context):
        pass
