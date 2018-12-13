import torch
import pandas as pd
import numpy as np
from itertools import product
from experiment_interface.utils import np_encode, np_decode
from experiment_interface.utils import extract_np_array

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

    def batch_to_df(self, *args, metadata):
        raise NotImplementedError()

    def summarize(self, df, mode='metric'):
        '''
        df: pd.DataFrame
            columns must be self column
        mode: str
            must be 'metric' or metric-speicific string
            when set to 'metric' must return a float
        '''
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


    def batch_to_df(self, predictions, labels, metadata):
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

    def summarize(self, df, mode='metric'):
        '''
        Input
        =====
            mode: 'conf_mat' | 'per_class_acc' | 'metric' 
                - confusion matrx ('conf_mat')
                    CxC pd.DataFrame, where both columns and index are self.columns.
                    row: predicted class; columns: groundtruth class
                - per-class accuracy ('per_class_acc')
                    length-C pd.Series. accuracy, or recall, per class 
                - mean accuracy ('metric')
                    average across classes of per-class accuracy
        '''

        conf_mat = self.confusion_mat(df)
        if mode == 'per_class_acc':
            return conf_mat
        C = conf_mat.values.astype(np.float32)
        per_class_acc = np.diag(C) / np.sum(C, axis=0)
        if mode == 'per_class_acc':
            return pd.Series(per_class_acc, index=self.columns)
        assert mode == 'metric'
        mean_accuracy = np.mean(per_class_acc)

        return mean_accuracy


class LossMetric(Metric):
    '''
    Assumption:
        loss_fn(reduction='mean') computes mean loss over the batch.
        loss_fn(reduction='none') does not perform the reduction.

    '''

    def __init__(self, loss_module):
        self.loss_fn = loss_module(reduction='none')
        self.type = 'loss'

    @property
    def larger_is_better(self):
        return False

    @property
    def columns(self):
        return ['image_id', 'loss']

    def batch_to_df(self, *loss_fn_args, metadata):

        num_examples = loss_fn_args[0].shape[0]
        elementwise_loss = self.loss_fn(*loss_fn_args)

        dikt = {
            'image_id': metadata['img_ids'],
            'loss': elementwise_loss.data.cpu().numpy(),
        }

        for arg_idx, arg in enumerate(loss_fn_args):
            _arg = arg.data.cpu().numpy()
            # dtype = _arg.dtype
            # shape = _arg[0].shape

            encoded_args, encoded_shapes, encoded_dtypes = list(zip(*[np_encode(x) for x in _arg]))

            dikt.update({
                'arg%d' % arg_idx: encoded_args,
                'arg%d_shape' % arg_idx: encoded_shapes,
                'arg%d_dtype' % arg_idx: encoded_dtypes,
                })


        df = pd.DataFrame(dikt)

        extracted_args0 = extract_np_array(df, 'arg0')
        extracted_args1 = extract_np_array(df, 'arg1')

        return df

    def summarize(self, df):
        return df['loss'].mean()

class PrecisionRecall(Metric):

    def __init__(self):
        raise NotImplementedError() # TODO

    @property
    def columns(self):
        return ['image_id', 
                'x1', 'y1', 'x2', 'y2', 
                'predicted_class', 
                'matched_gt_x1', 'matched_gt_y1', 'matched_gt_x2', 'matched_gt_y2']
