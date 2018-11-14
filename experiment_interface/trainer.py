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

# logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S')

logging.addLevelName(logging.INFO, '')

formatter = logging.Formatter(
    fmt='%(levelname)s %(asctime)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

class TrainContext():

    def __init__(self):
        '''
        A hook can set 'exit_loop' to True to terimnate the training loop.
        '''
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
        num_workers = None,
        hooks = [],
        log_file = None,
        ):

        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.net =net 
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.hooks = hooks
        self.log_file = log_file

    def setup_logger(self):
        logger = logging.getLogger('train_logger')
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if self.log_file is not None:
            fh = logging.FileHandler(filename=self.log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger


    def run(self):

        logger = self.setup_logger()

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

        context = TrainContext()

        for hook in self.hooks:
            hook.before_loop(context)

        # training loop
        while not context.exit_loop:

            for images, labels in train_data_loader:
                context.inc_step()

                for hook in self.hooks:
                    hook.before_step(context)

                images, labels = images.to(device), labels.to(device)
                prediction = net(images)
                loss = self.loss_fn(prediction, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logger.info('step=%d | loss=%.4f' % (context.step, loss.detach()))

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
