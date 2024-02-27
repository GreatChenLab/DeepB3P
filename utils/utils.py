# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : utils.py

import random
import numpy as np
from sklearn import metrics
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed, logger):
    logger.info('set seed {} for everything'.format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def cal_performance(y_true, y_pred, y_prob, logger, logging_=True):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    acc = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    auc = metrics.auc(fpr, tpr)
    if logging_:
        pred_num = len(y_pred)
        pred_1 = np.sum(y_pred, dtype=np.int32)
        true_1 = np.sum(y_true, dtype=np.int32)
        logger.info("tn={0}, fp={1}, fn={2}, tp={3}".format(tn, fp, fn, tp))
        logger.info("y_pred: 0={0} | 1={1}".format(pred_num - pred_1, pred_1))
        logger.info("y_true: 0={0} | 1={1}".format(pred_num - true_1, true_1))
        logger.info("auc={0:.4f}|sn={1:.4f}|sp={2:.4f}|acc={3:.4f}|mcc={4:.4f}".format(auc, sn, sp, acc, mcc))
    return (auc, sn, sp, acc, mcc)

class SeqDataset(Dataset):
    def __init__(self, x, y, keep_one_hot=False):
        if keep_one_hot:
            self.x_data = F.one_hot(torch.tensor(x))
        else:
            self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]

def get_dataloader(dataset, batch_size, shuffle=True, drop_last=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)