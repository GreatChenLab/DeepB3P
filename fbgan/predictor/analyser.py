# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : analyser.py

from threading import Thread
import pandas as pd

from fbgan.predictor.blast import get_local_blast_results
from fbgan.predictor.s_bp3pred import get_b3pred
from fbgan.predictor.s_BBPpred import get_BBBPpred
from fbgan.predictor.BBBPpredict import get_BBBPpredict


def run_blast(seqs, dbpath, logger):
    pos_db = dbpath / 'pos'
    neg_db = dbpath / 'neg'
    pos = get_local_blast_results(dbpath, pos_db, seqs)
    logger.info('pos blast results:\n{0}'.format(pos))
    neg = get_local_blast_results(dbpath, neg_db, seqs)
    logger.info('neg blast results:\n{0}'.format(neg))

    blast_pos = []
    for seq, value in pos.items():
        v_neg = neg.get(seq, None)
        if v_neg:
            if float(value) > float(v_neg):
                blast_pos.append(seq)
        else:
            blast_pos.append(seq)
    return blast_pos

def run_predictor(seqs, logger, predictor_path):
    if len(seqs) == 0:
        return pd.DataFrame()
    seqs_list = []
    for i in range(len(seqs)):
        id = f'>{i}'
        seq = seqs[i]
        seqs_list.append(id)
        seqs_list.append(seq)

    seqs_str = '\r\n'.join(seqs_list)

    threadPost = []
    data = {}

    b3pred = Thread(target=get_b3pred, args=(seqs, data, logger, predictor_path))
    b3pred.start()
    threadPost.append(b3pred)

    BBBPpred = Thread(target=get_BBBPpred, args=(seqs, data, logger, predictor_path))
    BBBPpred.start()
    threadPost.append(BBBPpred)

    BBBPpredict = Thread(target=get_BBBPpredict, args=(seqs_str, data, logger, predictor_path))
    BBBPpredict.start()
    threadPost.append(BBBPpredict)

    for thread in threadPost:
        thread.join()

    b3pred = data.get('b3pred')
    BBBPpred = data.get('BBBPpred')
    BBBPpredict = data.get('BBBPpredict')

    df = pd.merge(b3pred, BBBPpred, on='peptides')
    df = pd.merge(df, BBBPpredict, on='peptides')
    df['means'] = df.iloc[:,2:].mean(axis=1).round(2)
    df.sort_values(by='means', ascending=False, inplace=True)
    return df

def run_analyser(seqs, logger, dbpath, predictor_path, is_blast, is_predict):
    logger.info('begin run analyser with blast {0} and predict {1}'.format(is_blast, is_predict))
    if len(seqs) == 0:
        logger.info('no sequences to runing')
        return pd.DataFrame()
    if is_blast and is_predict:
        seqs = run_blast(seqs, dbpath, logger)
        return run_predictor(seqs, logger, predictor_path)
    if is_blast:
        seqs = run_blast(seqs, dbpath, logger)
        df = pd.DataFrame(columns=['peptides', 'prob_BBBPpred', 'prob_b3pred', 'prob_BBBPpredict', 'means'])
        df.peptides = seqs
        df.fillna(1, inplace=True)
        return df

    return run_predictor(seqs, logger, predictor_path)


if __name__ == '__main__':
    seqs = ['VLGGGSALLRSIPA','IGSENSEKTTMP','FLPLLAASFACTVTKKC']
    print('run test for analyser')
