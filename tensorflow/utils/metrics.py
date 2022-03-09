import sys
import numpy as np
from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    
    @property
    def avg(self):
        return self.sum / self.count

class Metrics(dict):
    def __init__(self, *args, scale=1, order=['mIoU', 'OA', 'mACC'], **kwargs):
        super(Metrics, self).__init__(*args, **kwargs)
        self.scale = scale
        self.order = [order] if isinstance(order, str) else list(order)  # the importance rank of metrics - main key = order[0]
    # def __missing__(self, key):
    #     return None

    # Comparison
    # ------------------------------------------------------------------------------------------------------------------

    def _is_valid(self, other, raise_invalid=True):
        if self.order[0] not in other:
            if raise_invalid:
                raise ValueError(f'missing main key - {self.order[0]}, in order {self.order}')
            return False
        return True

    def __eq__(self, other):  # care only the main key
        self._is_valid(self)
        self._is_valid(other)
        return self[self.order[0]] == other[self.order[0]]

    def __gt__(self, other):
        self._is_valid(self)
        self._is_valid(other)
        for k in self.order:
            if k not in self:  # skip if not available
                continue
            if k not in other or self[k] > other[k]:  # True if more completed
                return True
            elif self[k] < other[k]:
                return False

        # all equal (at least for main key)
        return False

    # Pretty print
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def scalar_str(self):
        scalar_m = [k for k in ['mIoU', 'OA', 'mACC'] if k in self and self[k]]
        s = ''.join([f'{k}={self[k]/self.scale*100:<6.2f}' for k in scalar_m])
        return s
    @property
    def list_str(self):
        list_m = [k for k in ['IoUs'] if k in self and self[k] is not None]
        s = []
        for k in list_m:
            m = self.list_to_line(k)
            s += [m]
        s = ' | '.join(s)
        return s
    @property
    def final_str(self):
        s = str(self)
        s = ['-' * len(s), s, '-' * len(s)]
        if 'ACCs' in self:
            s = ['ACCs = ' + self.list_to_line('ACCs')] + s
        return '\n'.join(s)
 
    def print(self, full=True, conf=True):
        s = self.full() if full else self.final_str
        if conf and 'conf' in self:
            conf = self['conf']
            assert np.issubdtype(conf.dtype, np.integer)
            with np.printoptions(linewidth=sys.maxsize, precision=3):
                print(self['conf'])
        print(s)

    def full(self, get_list=False):
        # separate line print each group of metrics
        scalar_m = [k for k in ['OA', 'mACC', 'mIoU'] if k in self and self[k]]
        name_d = {'IoUs': 'mIoU', 'ACCs':'mACC'}  # list_m -> scalar_m

        str_d = {k: f'{k}={self[k]/self.scale*100 if self[k] < 1 else self[k]:<6.2f}' for k in scalar_m}  # scalar_m -> str
        for k_list, k_scalar in name_d.items():
            str_d[k_scalar] += ' | ' + self.list_to_line(k_list)

        max_len = max(len(v) for v in str_d.values())
        s = ['-' * max_len, *[v for v in str_d.values()], '-' * max_len]
        s = s if get_list else '\n'.join(s)
        return s

    def __repr__(self):
        return ' | '.join([k for k in [self.scalar_str, self.list_str] if k])

    def list_to_line(self, k):
        l = k if isinstance(k, list) else self[k] if k in self else None
        m = ' '.join([f'{i/self.scale*100:<5.2f}' if i < 1 else f'{i:<5.2f}' for i in l]) if l is not None else ''
        return m

def metrics_from_confusions(confusions, proportions=None):
    """
    Computes IoU from confusion matrices.
    Args:
        confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes; gt (row) x pred (col).
    """

    confusions = confusions.astype(np.float32)
    if proportions is not None:
        # Balance with real proportions
        confusions *= np.expand_dims(proportions.astype(np.float32) / (confusions.sum(axis=-1) + 1e-6), axis=-1)

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    # Compute Accuracy
    OA = np.sum(TP, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)
    mACC = np.mean(TP / TP_plus_FN)
    m = {
        'mIoU': IoU.mean(),
        'mACC': mACC,
        'OA': OA,
        'IoUs': IoU,
        'ACCs': TP / TP_plus_FN,
    }
    m = Metrics(m)
    return m


def metrics_from_result(preds, labels, num_classes, label_to_idx=None, proportions=None):
    """
    list of pred-label
    """
    conf = 0
    num_classes = np.arange(num_classes) if isinstance(num_classes, int) else list(num_classes)
    for cur_pred, cur_label in zip(preds, labels):
        if len(cur_pred.shape) > 1:  # prob matrix
            cur_pred = np.argmax(cur_pred, axis=-1).astype(np.int)
        if label_to_idx is not None:  # match to the preds
            cur_label = label_to_idx[cur_label].astype(np.int)
            if np.any(cur_label < 0):  # potential invalid label position (would be ignored by specifying labels anyway)
                valid_mask = cur_label >= 0
                cur_pred = cur_pred[valid_mask]
                cur_label = cur_label[valid_mask]
        conf += confusion_matrix(cur_label, cur_pred, labels=num_classes)

    m = metrics_from_confusions(conf, proportions=proportions)
    m['conf'] = conf
    return m
