# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : s_BBPpred.py

# source code can be find in https://github.com/xialab-ahu/BBPpred
# modify more pretty
import re
import joblib
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
from utils.utils import *

def AAC(seqs):
    BBBAAC = "HKPQR"
    encodings = []
    header = ['#Name']
    for i in BBBAAC:
        header.append(i)
    encodings.append(header)

    for i in seqs:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in BBBAAC:
            count[key] = count[key] / len(sequence)
        code = [name]
        for aa in BBBAAC:
            code.append(count[aa])
        encodings.append(code)
    return encodings

def GAAC(seqs):
    group = {'alphatic': 'GAVLMI',
             'aromatic': 'FYW',
             'postivecharge': 'KRH',
             'negativecharge': 'DE',
             'uncharge': 'STCPNQ'}

    groupKey = group.keys()

    encodings1 = []
    header = []
    for key in groupKey:
        header.append(key)
    encodings1.append(header)

    for i in seqs:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        count = Counter(sequence)
        myDict = {}
        for key in groupKey:
            for aa in group[key]:
                myDict[key] = myDict.get(key, 0) + count[aa]

        for key in groupKey:
            code.append(myDict[key]/len(sequence))
        encodings1.append(code)

    return encodings1

def protein_length(seqs):
    encodings2 = []
    header = ["length"]
    encodings2.append(header)
    for i in seqs:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        length=len(sequence)
        Norlen=(length-5)/(82-5)
        code.append(Norlen)
        encodings2.append(code)
    return encodings2

def molecular_weight(seqs):
    #seq_new=seq.replace('X','').replace('B','')
    encodings3 = []
    header = ["Weight"]
    encodings3.append(header)
    for i in seqs:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        analysed_seq = ProteinAnalysis(sequence)
        analysed_seq.monoisotopic = True
        mw = analysed_seq.molecular_weight()
        Normw=(mw-513.222346)/(9577.017286-513.222346)
        code.append(Normw)
        encodings3.append(code)
    return(encodings3)

def get_BBBPpred(seqs, df, logger, predictor_path):
    logger.info("begin run BBBPpred")
    seq_list = []
    for i, v in enumerate(seqs):
        seq_list.append([i, v])

    myFun = "AAC(seqs)"
    myFun1 = "GAAC(seqs)"
    myFun2 = "protein_length(seqs)"
    myFun3 = "molecular_weight(seqs)"

    encodings = eval(myFun)
    encodings1 = eval(myFun1)
    encodings2 = eval(myFun2)
    encodings3 = eval(myFun3)
    encodings = ziped = list(
        map(lambda x: x[0] + x[1] + x[2] + x[3], zip(encodings, encodings1, encodings2, encodings3)))
    model = predictor_path / 'm_BBBPpred'
    logger.info(model)
    RF = joblib.load(model)
    PreData = pd.DataFrame(encodings[1:])
    PreDataX = PreData.drop([0], axis=1)
    result = RF.predict_proba(PreDataX)[:, -1]
    df1 = pd.DataFrame(columns=['peptides', 'prob_BBBPpred'])
    df1['peptides'] = seqs
    df1['prob_BBBPpred'] = result
    df['BBBPpred'] = df1
    logger.info(f'end for BBBPpred: \n {df1}')
    return None
