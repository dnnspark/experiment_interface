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
import os
from experiment_interface.hooks import ValidationHook, StopAtStep, ScalarRecorder, VisdomRunner
from experiment_interface.hooks.scalar_recorder import Row
from experiment_interface.hooks.viz_runner import plot_trainval_loss
from experiment_interface.logger import get_train_logger
from experiment_interface.evaluator.metrics import Metric

class TrainContext():

    def __init__(self, trainer, debug):
        '''
        A hook can set 'exit_loop' to True to terimnate the training loop.
        '''
        self.trainer = trainer
        self.debug = debug

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
        log_file='train.log',
        scalar_record_file = 'train_record.csv',
        max_step = None,
        log_interval = 1,
        num_workers = None,
        hooks = [],
        val_dataset = None,
        val_interval = None,
        ):

        self.logger = logger = get_train_logger(os.path.join(result_dir, log_file))

        self.use_cuda = use_cuda = torch.cuda.is_available()
        if use_cuda:
            num_gpus = torch.cuda.device_count()
            assert num_gpus > 0
            logger.info('CUDA device count = %d' % num_gpus)
        else:
            self.num_gpus = 0

        device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
        # move the net to a gpu, and setup replicas for using multiple gpus;
        # no change needed for cpu mode.
        net = net.to(device)
        self.net = torch.nn.DataParallel(net)

        self.train_dataset = train_dataset
        self.batch_size = batch_size

        self.loss_fn = loss_fn
        self.optimizer = optimizer # TODO: implement weight update schedule 
        self.result_dir = result_dir
        self.log_interval = log_interval

        if use_cuda and num_workers is None:
            raise ValueError('\'num_workers\' must be int, if cuda is available.')

        if not use_cuda and (num_workers is not None and num_workers > 0):
            logger.warning('\'num_workers=%d\' is ignored and set to zero, because use_cuda is False.' % num_workers)
            num_workers = 0

        self.num_workers = num_workers

        self.hooks = hooks

        if val_dataset is not None:

            if val_interval is None:
                raise ValueError('\'val_interval\' must be not None, if val_dataset is not None.')

            if isinstance(val_dataset, torch.utils.data.Dataset):
                name = 'val_loss'
                dataset = val_dataset 
                # predict_fn, metric = None, None
            elif len(val_dataset) == 2 and \
                isinstance(val_dataset[0], str) and isinstance(val_dataset[1], torch.utils.data.Dataset):
                name, dataset = val_dataset 
            else:
                raise ValueError('Invalid format for \'val_dataset\'.')

            val_hook = ValidationHook(dataset, val_interval, name, save_best=True)


            self.hooks.append( val_hook )

        if max_step is not None:
            self.hooks.append( StopAtStep(max_step) )

        if not scalar_record_file.endswith('.csv'):
            raise ValueError('Scalar log file must have .csv extension.')

        scalar_record_file = os.path.join(result_dir, scalar_record_file)
        scalar_recorder = ScalarRecorder(scalar_record_file)
        self.scalar_record_file = scalar_record_file
        self.hooks.append(scalar_recorder)
        self.scalar_recorder = scalar_recorder

        trainloss_viz_runner = VisdomRunner(plot_fn=plot_trainval_loss)
        self.hooks.append(trainloss_viz_runner)


    def register_hook(self, hook, prepend=True):
        if prepend:
            self.hooks = [hook] + self.hooks
        else:
            self.hooks = self.hooks + [hook]

    def register_validation_hook(self, hook, prepend=True):
        hook.set_cache_dir(os.path.join(self.result_dir, 'val'))
        self.register_hook(hook, prepend)

    def run(self, debug=False):

        logger = self.logger
        if debug:
            self.log_interval = 1

        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, 
            num_workers=self.num_workers, pin_memory=self.use_cuda)

        self.net.train()
        net = self.net

        context = TrainContext(self, debug=debug)

        for hook in self.hooks:
            hook.before_loop(context)

        # training loop
        while not context.exit_loop:

            for batch in train_data_loader:
                # Assumptions: 
                # - batch is a dict key'ed by ('inputs', 'labels', 'metadata')
                # - batch['inputs'] is either a Tensor or a list/tuple of Tensor
                # - batch['labels'] is either a Tensor or a list/tuple of Tensor
                # - batch['metadata'] is a dict of str.

                for hook in self.hooks:
                    hook.before_step(context)

                inputs, labels, metadata = batch['inputs'], batch['labels'], batch['metadata']

                if isinstance(inputs, torch.Tensor):
                    inputs = [inputs]
                if isinstance(labels, torch.Tensor):
                    labels = [labels]

                if self.use_cuda:
                    # move input to a gpu
                    device = torch.device('cuda:0')
                    inputs = [x.to(device) for x in inputs]
                    labels = [x.to(device) for x in labels]

                context.inc_step()

                net_outputs = net(*inputs)
                if isinstance(net_outputs, torch.Tensor):
                    net_outputs = [net_outputs]
                losses = self.loss_fn(*net_outputs, *labels)

                # Assumption:
                # - self.loss_fn() returns either a 0-dim (scalar) Tensor, or a list/tuple of the following form
                # - [total_loss, (loss_name_1, loss1), (loss_name_2, loss2), ...]
                if isinstance(losses, list) or isinstance(losses, tuple):
                    total_loss, other_losses = losses[0], losses[1:]
                else:
                    if losses.dim() != 0 :
                        raise ValueError('loss must be a scalar Tensor.')
                    total_loss = losses
                    other_losses = []

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if context.step % self.log_interval == 0:
                    _total_loss = total_loss.detach().cpu().numpy()
                    logger.info('step=%d | total_loss=%.4f' % (context.step, _total_loss))
                    if self.scalar_recorder is not None:
                        self.scalar_recorder.append( Row(context.step, 'batch_loss', _total_loss) )

                for hook in self.hooks:
                    hook.after_step(context)

                if context.exit_loop:
                    break;

        for hook in self.hooks:
            hook.after_loop(context)


