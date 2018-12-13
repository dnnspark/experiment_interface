import os
import numpy as np
import torch
import pandas as pd
from experiment_interface.logger import get_train_logger, get_test_logger

class Evaluator():

    def __init__(self,
        net,
        test_dataset,
        batch_size,
        predict_fn,
        metric,
        num_workers,
        result_dir=None,
        record_file=None,
        log_file=None,
        pretrained_params_file=None,
        is_validating=False,
        ):

        if is_validating:
            logger = get_train_logger()
        else:
            if log_file is not None:
                if result_dir is None:
                    raise ValueError('\'result_dir\' must be not None, if \'log_file\' it not None. ')
                logger = get_test_logger(os.path.join(result_dir, log_file))
            else:
                logger = get_test_logger(None)
        self.logger = logger

        self.use_cuda = use_cuda = torch.cuda.is_available()
        if use_cuda:
            num_gpus = torch.cuda.device_count()
            assert num_gpus > 0
            logger.info('CUDA device count = %d' % num_gpus)
        else:
            self.num_gpus = 0

        if is_validating:
            # use input net as it is.
            self.net = net
        else:
            device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
            # move the net to a gpu, and setup replicas for using multiple gpus;
            # no change needed for cpu mode.
            net = net.to(device)
            self.net = torch.nn.DataParallel(net)

        if pretrained_params_file is not None:
            # (TODO) load the pretrained model
            import pdb; pdb.set_trace()

        self.test_dataset = test_dataset
        self.batch_size = batch_size

        self.predict_fn = predict_fn
        self.metric = metric

        if use_cuda and num_workers is None:
            raise ValueError('\'num_workers\' must be int, if cuda is available.')

        if not use_cuda and (num_workers is not None and num_workers>0):
            logger.warning('\'num_workers=%d\' is ignored and set to zero, because use_cuda is False.' % num_workers)
            num_workers = 0

        self.num_workers = num_workers

        if record_file is not None:
            if result_dir is None:
                raise ValueError('\'result_dir\' must be not None, if \'record_file\' is not None. ')
            record_file = os.path.join(result_dir, record_file)
        self.record_file = record_file 

    def set_record_file(self, record_file):
        if self.record_file is not None:
            raise ValueError('\'record_file\' is not None, and overwriting is not allowed.')

        if not record_file.endswith('.csv'):
            raise ValueError('\'record_file\' must have .csv extension.')

        self.record_file = record_file


    def run(self):

        logger = self.logger

        test_data_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False, 
            num_workers=self.num_workers, pin_memory=self.use_cuda)

        self.net.eval()
        net = self.net

        # class bbox img_file matched_groundtruth score 

        # self.metric.initialize()

        df = pd.DataFrame(columns=self.metric.columns)

        for batch in test_data_loader:
            # Assumptions: 
            # - batch is a dict key'ed by ('inputs', 'labels', 'metadata')
            # - batch['inputs'] is either a Tensor or a list/tuple of Tensor
            # - batch['labels'] is either a Tensor or a list/tuple of Tensor
            # - batch['metadata'] is a dict of str.
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

            with torch.no_grad():
                # TODO: this is probably not enougth to turn off requires_grad
                net_outputs = net(*inputs)
            if isinstance(net_outputs, torch.Tensor):
                net_outputs = [net_outputs]

            predictions = self.predict_fn(*net_outputs)
            if isinstance(predictions, torch.Tensor):
                predictions = [predictions]

            _df = self.metric.batch_to_df(*predictions, *labels, metadata=metadata) 
            df = pd.DataFrame.append(df, _df, ignore_index=True, sort=False)

        if self.record_file is not None:
            logger.info("Writing %s." % self.record_file)
            with open(self.record_file, 'w') as f:
                df.to_csv(f)

        eval_metric = self.metric.summarize(df)
        return eval_metric
