import torch
import numpy as np

class Metric():
    '''
    '''
    def initialize(self):
        raise NotImplementedError()

    def process_batch_result(self, *args, metadata):
        '''
        This takes a batch of predictions and its groundtruth,
        and update its internal state.
        '''
        raise NotImplementedError()

    def summarize(self, result_file=None):
        raise NotImplementedError()

class ClassificationAccuracy(Metric):

    def __init__(self,
        category_names
        ):
        self.num_categories = len(category_names)


    def initialize(self):

        self.num_labels = np.zeros(self.num_categories)
        self.num_matched = np.zeros(self.num_categories)

    def process_batch_result(self, predictions, labels, metadata):
        '''
        '''

        labels = labels.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()

        label_count = np.bincount( labels, minlength=self.num_categories)
        matched = predictions == labels
        match_count = np.bincount( labels, minlength=self.num_categories, weights=np.float32(matched))
        self.num_labels += label_count
        self.num_matched += match_count

    def summarize(self, result_file=None):
        accuracy = self.num_matched / self.num_labels
        return np.mean(accuracy)

class LossMetric(Metric):
    '''
    Assumption:
        loss_fn computes mean loss over the batch.
    '''

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def initialize(self):

        self.total_loss = 0
        self.num_examples = 0

    def process_batch_result(self, *loss_fn_args, metadata):

        num_examples = loss_fn_args[0].shape[0]
        loss = self.loss_fn(*loss_fn_args)

        self.total_loss += num_examples * loss
        self.num_examples += num_examples

    def summarize(self, result_file=None):
        return self.total_loss / self.num_examples

class PrecisionRecall(Metric):

    def __init__(self):
        raise NotImplementedError() # TODO
