# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/9
# email  : tangqiang.0701@gmail.com
# file   : validation.py

from model.deepb3p import DeepB3P
from utils.utils import *
from utils.amino_acid import *

def get_models(params, logger):
    model_list = []
    for i in range(1, params.kFold+1):
        model_file = params.model_file / f'deepb3p_{i}.pth'
        model = DeepB3P(params, logger)
        model.model.reset_parameters()
        model.load_model(directory=model_file)
        model_list.append(model)
    return model_list


def cv_predict(params, dataloader, logger):
    model_list = get_models(params, logger)
    y_prob_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
    y_pred_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
    auc_list, sn_list, sp_list, acc_list, mcc_list = [], [], [], [], []
    for idx, model in enumerate(model_list):
        ys_train, loss_list, metrics_train, time_epoch = model.valid_epoch(dataloader)
        enc_att, dec_att, dec_enc_att = model.model.get_atts()
        y_true_list, y_pred_list, y_prob_list = ys_train
        y_prob_df[idx+1] = y_prob_list
        y_pred_df[idx+1] = y_pred_list
        auc, sn, sp, acc, mcc = metrics_train
        auc_list.append(auc)
        sn_list.append(sn)
        sp_list.append(sp)
        acc_list.append(acc)
        mcc_list.append(mcc)
    return auc_list, sn_list, sp_list, acc_list, mcc_list, y_prob_df, y_pred_df, y_true_list

def predict(params, logger):
    pos_test = './data/pos_test.fasta'
    neg_test = './data/neg_test.fasta'
    pos_test_fea, pos_test_label = fasta_to_numpy(pos_test)
    neg_test_fea, neg_test_label = fasta_to_numpy(neg_test, label=0)
    logger.info('read pos test seqs: {0}'.format(len(pos_test_fea)))
    logger.info('read neg test seqs: {0}'.format(len(neg_test_fea)))

    test_feas = np.concatenate((pos_test_fea, neg_test_fea), axis=0)
    test_labels = np.concatenate((pos_test_label, neg_test_label), axis=0)

    test_dataset = SeqDataset(test_feas, test_labels)
    dataloader = DataLoader(test_dataset, batch_size=len(test_labels))
    auc_list, sn_list, sp_list, acc_list, mcc_list, prob_df, pred_df, y_true = cv_predict(params, dataloader, logger)
    auc = sum(auc_list)/len(auc_list)
    sn = sum(sn_list) / len(sn_list)
    sp = sum(sp_list) / len(sp_list)
    acc = sum(acc_list) / len(acc_list)
    mcc = sum(mcc_list) / len(mcc_list)
    logger.info('Average result for {}-fold'.format(params.kFold))
    row_first = ['Average', 'auc', 'sn', 'sp', 'acc', 'mcc']
    metrics_list = [auc, sn, sp, acc, mcc]
    logger.info(''.join(f'{item:<12}' for item in row_first))
    logger.info(f'%-12s' % 'Average1' + ''.join(f'{key:<12.3f}' for key in metrics_list))

    avg_prob = prob_df.mean(axis=1)
    avg_prob_pred = transfer(avg_prob, 0.5)
    metrics_train_avg = cal_performance(y_true, avg_prob_pred, avg_prob, logger, logging_=True)
    logger.info(''.join(f'{item:<12}' for item in row_first))
    logger.info(f'%-12s' % 'Average' + ''.join(f'{key:<12.3f}' for key in metrics_train_avg))

    vote_pred = pred_df.mean(axis=1)
    vote_pred = transfer(vote_pred, 0.5)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, vote_pred, labels=[0, 1]).ravel().tolist()
    acc = metrics.accuracy_score(y_true, vote_pred)
    mcc = metrics.matthews_corrcoef(y_true, vote_pred)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    vote_row = ['vote', 'sn', 'sp', 'acc', 'mcc']
    metrics_train = [sn, sp, acc, mcc]
    logger.info(''.join(f'{item:<12}' for item in vote_row))
    logger.info(f'%-12s' % 'vote' + ''.join(f'{key:<12.3f}' for key in metrics_train))
    auc, sn, sp, acc, mcc = metrics_train_avg
    return auc, sn, sp, acc, mcc
