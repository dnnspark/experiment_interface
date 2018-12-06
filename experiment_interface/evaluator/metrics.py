import torch
import pandas as pd
import numpy as np
from itertools import product

class Metric():
    '''
    '''

    @property
    def larger_is_better(self):
        '''
        True or False
        '''
        raise NotImplementedError()

    @property
    def columns(self):
        raise NotImplementedError()

    # def initialize(self):
    #     raise NotImplementedError()

    def process_batch_result(self, *args, metadata):
        '''
        This takes a batch of predictions and its groundtruth,
        and update its internal state.

        Return
        ======
            record: pandas.Dataframe
                must have the self.columns as columns
        '''
        raise NotImplementedError()

    def summarize(self, df):
        raise NotImplementedError()

class ClassificationAccuracy(Metric):

    def __init__(self,
        category_names
        ):
        self.category_names = category_names
        self.num_categories = len(category_names)

    @property
    def larger_is_better(self):
        return True

    @property
    def columns(self):
        return ['image_id', 'predicted_class_idx', 'groundtruth_class_idx', 'predicted_class', 'groundtruth_class']

    # def initialize(self):

    #     self.num_labels = np.zeros(self.num_categories)
    #     self.num_matched = np.zeros(self.num_categories)

    def process_batch_result(self, predictions, labels, metadata):
        '''
        '''
        assert len(predictions) == len(labels) == len(metadata['img_ids'])

        labels = labels.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()

        predicted_class = [self.category_names[p] for p in predictions]
        groundtruth_class = [self.category_names[p] for p in labels]

        df = pd.DataFrame({
            'image_id': metadata['img_ids'],
            'predicted_class_idx': predictions,
            'groundtruth_class_idx': labels,
            'predicted_class': predicted_class,
            'groundtruth_class': groundtruth_class,
            })

        return df
        # label_count = np.bincount( labels, minlength=self.num_categories)
        # matched = predictions == labels
        # match_count = np.bincount( labels, minlength=self.num_categories, weights=np.float32(matched))
        # self.num_labels += label_count
        # self.num_matched += match_count

    def confusion_mat(self, df):

        # indices = range(self.num_categories)
        idx_xproduct = product(self.category_names, self.category_names)
        I = pd.MultiIndex.from_tuples(idx_xproduct, names=['predicted_class', 'groundtruth_class'])
        hist = pd.Series(index=I).fillna(0).astype(np.int64)

        hist_ = df.groupby(['predicted_class', 'groundtruth_class']).size()

        hist.update(hist_)
        hist = hist.reset_index(name='count')
        conf_mat = hist.pivot(index='predicted_class', columns='groundtruth_class', values='count')

        conf_mat.index.name = 'predicted'
        conf_mat.columns.name = 'groundtruth'

        return conf_mat

    def summarize(self, df):
        '''
        From rich to succinct:
            - Confusion matrix
            - per-class accuracy
            - mean accuracy
        '''

        conf_mat = self.confusion_mat(df)
        import pdb; pdb.set_trace()

        # product(range(self.num_categories), range(self.num_categories))

        # count = df.groupby(['predicted_class_idx', 'groundtruth_class_idx']).size() # pd.Series
        # count = count.reset_index(name='count') # pd.DataFrame

        # accuracy = (df['predicted_class_idx'] == df['groundtruth_class_idx']).astype(np.float32).mean()
        # accuracy = self.num_matched / self.num_labels
        # return np.mean(accuracy)

        return accuracy

class LossMetric(Metric):
    '''
    Assumption:
        loss_fn computes mean loss over the batch.
    '''

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    @property
    def larger_is_better(self):
        return False

    def initialize(self):

        self.total_loss = 0
        self.num_examples = 0

    def process_batch_result(self, *loss_fn_args, metadata):

        num_examples = loss_fn_args[0].shape[0]
        loss = self.loss_fn(*loss_fn_args)

        self.total_loss += num_examples * loss
        self.num_examples += num_examples

    def summarize(self, df):
        return self.total_loss / self.num_examples

class PrecisionRecall(Metric):

    def __init__(self):
        raise NotImplementedError() # TODO

    @property
    def columns(self):
        return ['image_id', 
                'x1', 'y1', 'x2', 'y2', 
                'predicted_class', 
                'matched_gt_x1', 'matched_gt_y1', 'matched_gt_x2', 'matched_gt_y2']
