# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : deepb3p.py

import os
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SubsetRandomSampler

from model.classifier_transformer2 import Classifier
from utils.utils import *

class DeepB3P():
    def __init__(self, params, logger):
        super(DeepB3P, self).__init__()
        self.seq_len = params.seq_len
        self.vocab_size = params.vocab_size
        self.d_model = params.d_model
        self.n_heads = params.n_heads
        self.d_k = params.d_k
        self.d_ff = params.d_ff
        self.n_layers=params.n_layers
        self.device = params.device
        self.dropout = params.drop
        self.bs = params.bs
        self.epochs = params.n_epochs
        self.lr = params.lr
        self.checkpoint = params.model_file
        self.l_mean = lambda l: sum(l) / len(l)
        self.logger = logger
        self.build_model()

    def build_model(self):
        self.logger.info("the device is {}".format(self.device))
        self.logger.info("Begining init the deepb3p model with the lr is: {}".format(self.lr))
        #seq_len, vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers, device
        self.model = Classifier(
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_k,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            drop=self.dropout,
            device=self.device
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-07)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger.info(f'the model:\n{self.model}')

    def save_model(self, kFlod):
        torch.save(self.model.eval().state_dict(), os.path.join(self.checkpoint, f'deepb3p_{kFlod}.pth'))

    def load_model(self, directory=None):
        if directory is None:
            directory = self.checkpoint + 'deepb3p_1.pth'
        if not os.path.exists(directory):
            self.logger.warning('Checkpoint not found! Starting from scratch.')
            return 0
        self.logger.info(f'Loading model from {directory}')
        self.model.load_state_dict(torch.load(directory))

    def train_epoch(self, data_loader, optimizer):
        self.model.train()
        y_true_list, y_prob_list, loss_list = [], [], []
        train_start = time.time()
        for feats, labels in tqdm(data_loader, mininterval=1, desc='Training Processing', leave=False):
            feats, labels = feats.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(feats)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            y_train = labels.cpu().detach().numpy()
            y_prob = outputs[:, 1].cpu().detach().numpy()
            loss_trian = loss.cpu().detach().numpy()
            y_true_list.extend(y_train)
            y_prob_list.extend(y_prob)
            loss_list.append(loss_trian)
        time_epoch = (time.time() - train_start) / 60
        y_pred_list = transfer(y_prob_list, 0.5)
        ys_train = (y_true_list, y_pred_list, y_prob_list)
        metrics_train = cal_performance(y_true_list, y_pred_list, y_prob_list, self.logger, logging_=True)
        return ys_train, loss_list, metrics_train, time_epoch

    def valid_epoch(self, data_loader):
        train_start = time.time()
        y_true_list, y_prob_list, loss_list = [], [], []
        with torch.no_grad():
            self.model.eval()
            for feats, labels in tqdm(data_loader, mininterval=1, desc='Validing Processing', leave=False):
                feats, labels = feats.to(self.device), labels.to(self.device)
                outputs = self.model(feats)
                loss = self.criterion(outputs, labels)
                y_train = labels.cpu().detach().numpy()
                y_prob = outputs[:, 1].cpu().detach().numpy()
                loss_trian = loss.cpu().detach().numpy()
                y_true_list.extend(y_train)
                y_prob_list.extend(y_prob)
                loss_list.append(loss_trian)
            time_epoch = (time.time() - train_start) / 60
            y_pred_list = transfer(y_prob_list, 0.5)
            ys_train = (y_true_list, y_pred_list, y_prob_list)
            metrics_train = cal_performance(y_true_list, y_pred_list, y_prob_list, self.logger, logging_=True)
            return ys_train, loss_list, metrics_train, time_epoch

    def cv_train(self, dataset, kFlod=5, earlyStop=5, seed=2023):
        splits = StratifiedKFold(n_splits=kFlod, shuffle=True, random_state=seed+self.vocab_size)
        fold_best = []
        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset[:][0], dataset[:][1])):
            set_seed(fold + 1, self.logger)
            self.model.reset_parameters()
            best_acc = 0.0
            self.logger.info(f'begin train with fold {fold + 1}')
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(dataset, batch_size=self.bs, sampler=train_sampler, drop_last=True)
            valid_loader = DataLoader(dataset, batch_size=self.bs, sampler=valid_sampler, drop_last=True)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-07)

            best_record = {'train_loss': 0, 'test_loss': 0, 'train_acc': 0, 'test_acc': 0, 'train_f': 0,
                           'test_f': 0, 'train_pre': 0, 'test_pre': 0, 'train_rec': 0, 'test_rec': 0,
                           'train_roc': 0, 'test_roc': 0}
            nobetter = 0
            for epoch in range(1, self.epochs + 1):
                # metrics --> auc, sn, sp, acc, mcc
                ys_train, loss_train, metrics_train, time_epoch = self.train_epoch(train_loader, optimizer)
                loss_train = np.mean(loss_train)
                self.logger.info(f'training: Epoch-{epoch}/{self.epochs} | loss={loss_train:.4f} | time={time_epoch:.4f} min')
                ys_valid, loss_valid, metrics_valid, time_epoch = self.valid_epoch(valid_loader)
                loss_valid = np.mean(loss_valid)
                self.logger.info(f'validing: Epoch-{epoch}/{self.epochs} | loss={loss_valid:.4f} | time={time_epoch:.4f} min')

                if best_acc < metrics_valid[3]:
                    nobetter = 0
                    best_acc = metrics_valid[3]
                    best_record['valid_loss'] = loss_valid
                    best_record['valid_auc'] = metrics_valid[0]
                    best_record['valid_sn'] = metrics_valid[1]
                    best_record['valid_sp'] = metrics_valid[2]
                    best_record['valid_acc'] = metrics_valid[3]
                    best_record['valid_mcc'] = metrics_valid[4]
                    best_record['train_loss'] = loss_train
                    best_record['train_auc'] = metrics_train[0]
                    best_record['train_sn'] = metrics_train[1]
                    best_record['train_sp'] = metrics_train[2]
                    best_record['train_acc'] = metrics_train[3]
                    best_record['train_mcc'] = metrics_train[4]
                    self.logger.info('Get a better model with acc {0:.4f}'.format(best_acc))
                    self.save_model(kFlod=fold + 1)
                else:
                    nobetter += 1
                    if nobetter >= earlyStop:
                        self.logger.info(f'validing acc has not improved for more '
                                     f'than {earlyStop} steps in epoch {epoch}, stop training')
                        break
            fold_best.append(best_record)
            self.logger.info(f'cv fold {kFlod} for fold {fold + 1} done')
            self.logger.info(f'Find best model, valid auc:{best_record["valid_auc"]:.3f},  '
                         f'sn:{best_record["valid_sn"]:.3f},  sp:{best_record["valid_sp"]:.3f}, '
                         f'acc:{best_record["valid_acc"]:.3f}, mcc:{best_record["valid_mcc"]:.3f}')
        self.logger.info('all folds are done')
        row_first = ['Fold', 'auc', 'sn', 'sp', 'acc', 'mcc']
        self.logger.info(''.join(f'{item:<12}' for item in row_first))
        metrics = ['valid_auc', 'valid_sn', 'valid_sp', 'valid_acc', 'valid_mcc']
        for idx, fold in enumerate(fold_best):
            self.logger.info(f'{idx + 1:<12}' + ''.join(
                f'{fold[key]:<12.3f}' for key in metrics
            ))
        avg = {}
        for item in metrics:
            avg[item] = 0
            for fold in fold_best:
                avg[item] += fold[item]
            avg[item] /= kFlod
        self.logger.info(f'%-12s' % 'Average' + ''.join(f'{avg[key]:<12.3f}' for key in metrics))
