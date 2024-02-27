# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : s_bp3pred.py

# source code can be find in https://webs.iiitd.edu.in/raghava/b3pred/Standalone.php
# modify more pretty
# need scikit-learn 0.21.3 or old

import math
import pandas as pd
import joblib
from collections import Counter
from itertools import product
import more_itertools
from utils.utils import *

def count_mers(seq, k=1):
    return Counter(("".join(mers) for mers in more_itertools.windowed(seq, k)))

std = list("ACDEFGHIKLMNPQRSTVWY")

def aac_comp(seqs):
    aac_title = ['AAC_' + one for one in std]
    df_acc = pd.DataFrame(columns=aac_title)
    for seq in seqs:
        result = []
        seq_len = len(seq)
        counter = count_mers(seq, 1)
        for i in std:
            comp = (counter.get(i, 0) / seq_len) * 100
            result.append(round(comp, 2))
        df = pd.DataFrame(columns=aac_title, data=[result])
        if df_acc.empty is True:
            df_acc = df
        else:
            df_acc = df_acc.append(df, ignore_index=True)
    df_acc['seqs'] = seqs
    return df_acc

def dpc_comp(seqs):
    dpc_list = [''.join(list(one)) for one in list(product(std, repeat=2))]
    dpc_title = ['DPC1_'+one for one in dpc_list]
    df_dpc = pd.DataFrame(columns=dpc_title)
    for seq in seqs:
        res = []
        counter = count_mers(seq, 2)
        n_mer = len(seq) - 1
        for one in dpc_list:
            comp = (counter.get(one, 0) / n_mer) * 100
            res.append(round(comp, 2))
        df = pd.DataFrame(columns=dpc_title, data=[res])
        if df_dpc.empty is True:
            df_dpc = df
        else:
            df_dpc = df_dpc.append(df, ignore_index=True)
    df_dpc['seqs'] = seqs
    return df_dpc

def tpc_comp(seqs):
    tpc_list = [''.join(list(one)) for one in list(product(std, repeat=3))]
    tpc_title = ['TPC_' + one for one in tpc_list]
    df_tpc = pd.DataFrame(columns=tpc_title)
    for seq in seqs:
        res = []
        counter = count_mers(seq, 3)
        n_mer = len(seq) - 2
        for one in tpc_list:
            comp = (counter.get(one, 0) / n_mer) * 100
            res.append(round(comp, 2))
        df = pd.DataFrame(columns=tpc_title, data=[res])
        if df_tpc.empty is True:
            df_tpc = df
        else:
            df_tpc = df_tpc.append(df, ignore_index=True)
    df_tpc['seqs'] = seqs
    return df_tpc

def apacc_1(seqs):
    lambdaval = 1
    w = 0.05
    data1 = pd.DataFrame({
        '#': ['Hydrophobicity', 'Hydrophilicity', 'SideChainMass'],
        'A':[0.62, -0.5, 15],
        'C':[0.29, -1, 47],
        'D':[-0.9, 3, 59],
        'E':[-0.74, 3, 73],
        'F':[1.19, -2.5, 91],
        'G':[0.48, 0, 1],
        'H':[-0.4, -0.5, 82],
        'I':[1.38, -1.8, 57],
        'K':[-1.5, 3, 73],
        'L':[1.06, -1.8, 57],
        'M':[0.64, -1.3, 75],
        'N':[-0.78, 0.2, 58],
        'P':[0.12, 0, 42],
        'Q':[-0.85, 0.2, 72],
        'R':[-2.53, 3, 101],
        'S':[-0.18, 0.3, 31],
        'T':[-0.05, -0.4, 45],
        'V':[1.08, -1.5, 43],
        'W':[0.81, -3.4, 130],
        'Y':[0.26, -2.3, 107]
    })
    aa = {}
    for i in range(len(std)):
        aa[std[i]] = i

    dd = []
    for i in range(0, 3):
        mean = sum(data1.iloc[i][1:]) / 20
        rr = math.sqrt(sum([(p - mean) ** 2 for p in data1.iloc[i][1:]]) / 20)
        dd.append([(p - mean) / rr for p in data1.iloc[i][1:]])
    zz = pd.DataFrame(dd)

    head = []
    for n in range(1, lambdaval + 1):
        for e in ('HB', 'HL', 'SC'):
            head.append(e + '_lam' + str(n))
    head = ['APAAC' + str(lambdaval) + '_' + sam for sam in head]

    ee = []
    df_apacc_1 = pd.DataFrame(columns=head)
    for seq in seqs:
        cc = []
        for n in range(1, lambdaval + 1):
            for b in range(0, len(zz)):
                cc.append(sum([zz.loc[b][aa[seq[p]]] * zz.loc[b][aa[seq[p + n]]] for p in range(len(seq) - n)]) / (
                            len(seq) - n))
        pseudo = [round((w * p) / (1 + w * sum(cc)), 4) for p in cc]
        df = pd.DataFrame(columns=head, data=[pseudo])
        if df_apacc_1.empty is True:
            df_apacc_1 = df
        else:
            df_apacc_1 = df_apacc_1.append(df, ignore_index=True)
    df_apacc_1['seqs'] = seqs
    return df_apacc_1

def apacc(df_apacc_1, df_aac):
    apacc_title = ['APAAC1_'+i for i in std]
    df_acc_title = df_aac.columns.tolist()
    df_acc_title.remove('seqs')
    rename = {}
    for i,j in zip(df_acc_title, apacc_title):
        rename[i] = j
    df_acc_copy = df_aac.rename(columns=rename)
    df = pd.merge(df_acc_copy, df_apacc_1, on='seqs')
    return df

def atc(seqs):
    atom = {
        'A': 'CHHHCHNHHCOOH',
        'R': 'HNCNHHNHCHHCHHCHHCHNHHCOOH',
        'N': 'HHNCOCHHCHNHHCOOH',
        'D': 'HOOCCHHCHNHHCOOH',
        'C': 'HSCHHCHNHHCOOH',
        'Q': 'HHNCOCHHCHHCHNHHCOOH',
        'E': 'HOOCCHHCHHCHNHHCOOH',
        'G': 'NHHCHHCOOH',
        'H': 'NHCHNCHCCHHCHNHHCOOH',
        'I': 'CHHHCHHCHCHHHCHNHHCOOH',
        'L': 'CHHHCHHHCHCHHCHNHHCOOH',
        'K': 'HHNCHHCHHCHHCHHCHNHHCOOH',
        'M': 'CHHHSCHHCHHCHNHHCOOH',
        'F': 'CCCCCCHHHHHCHHCHNHHCOOH',
        'P': 'NHCHHCHHCHHCHCOOH',
        'S': 'HOCHHCHNHHCOOH',
        'T': 'CHHHCHOHCHNHHCOOH',
        'W': 'CCCCCCHHHHHNHCHCCHHCHNHHCOOH',
        'Y': 'HOCCCCCCHHHHHCHHCHNHHCOOH',
        'V': 'CHHHCHHHCHCHNHHCOOH'}
    atomv = ['C', 'O', 'H', 'N', 'S']
    atc_title = ['ATC_C', 'ATC_H', 'ATC_N', 'ATC_O', 'ATC_S']
    atc_df = pd.DataFrame(columns=atc_title)
    for seq in seqs:
        atom_c = [0, 0, 0, 0, 0]
        for s in list(seq):
            value = atom.get(s)
            counter = Counter(value)
            for i in range(len(atom_c)):
                atom_c[i] += counter.get(atomv[i], 0)
        atom_sum = sum(atom_c)
        atom_c = [round((one/atom_sum)*100, 2) for one in atom_c]
        df = pd.DataFrame(columns=atc_title, data=[atom_c])
        if atc_df.empty:
            atc_df = df
        else:
            atc_df = atc_df.append(df, ignore_index=True)
    atc_df['seqs'] = seqs
    return atc_df

def bond(seqs):
    bonds = {
        #nBonds_tot,Hydrogen_bonds,nBondsS,nBondsD
        'G': [9,5,8,1],
        'S': [13,7,12,1],
        'A': [12, 7, 11, 1],
        'D': [15, 7, 13, 2],
        'N': [16, 8, 14, 2],
        'T': [16, 9, 15, 1],
        'P': [17, 9, 16, 1],
        'E': [18, 9, 16, 2],
        'V': [18, 11, 17, 1],
        'Q': [19, 10, 17, 2],
        'M': [19, 11, 18, 1],
        'H': [20, 9, 17, 3],
        'I': [21, 13, 20, 1],
        'Y': [24, 11, 20, 4],
        'L': [21, 13, 20, 1],
        'K': [23, 14, 22, 1],
        'W': [28, 12, 23, 5],
        'F': [23, 11, 19, 4],
        'C': [25, 12, 23, 2],
        'R': [25, 14, 23, 2]}
    bond_title = ['BTC_T', 'BTC_H', 'BTC_S', 'BTC_D']
    df_bond = pd.DataFrame(columns=bond_title)
    for seq in seqs:
        bond_list = [0, 0, 0, 0]
        for s in list(seq):
            value = bonds.get(s)
            for i in range(len(value)):
                bond_list[i] += value[i]
        df = pd.DataFrame(columns=bond_title, data=[bond_list])
        if df_bond.empty is True:
            df_bond = df
        else:
            df_bond = df_bond.append(df, ignore_index=True)
    df_bond['seqs'] = seqs
    return df_bond

def DDOR(seqs):
    ddor_title = ['DDR_'+one for one in std]
    df_ddor = pd.DataFrame(columns=ddor_title)
    for seq in seqs:
        seq_f = seq[::-1]
        zz2 = []
        for j in std:
            zz = ([pos for pos, char in enumerate(seq) if char == j])
            pp = ([pos for pos, char in enumerate(seq_f) if char == j])
            ss = []
            for i in range(0, (len(zz) - 1)):
                ss.append(zz[i + 1] - zz[i] - 1)
            if zz == []:
                ss = []
            else:
                ss.insert(0, zz[0])
                ss.insert(len(ss), pp[0])
            cc1 = (sum([e for e in ss]) + 1)
            cc = sum([e * e for e in ss])
            zz2.append(round(cc/cc1, 2))
        df = pd.DataFrame(columns=ddor_title, data=[zz2])
        if df_ddor.empty is True:
            df_ddor = df
        else:
            df_ddor = df_ddor.append(df, ignore_index=True)
    df_ddor['seqs'] = seqs
    return df_ddor

def ctd(seqs):
    ctd_title = []
    head1 = ['CeTD_HB', 'CeTD_VW', 'CeTD_PO', 'CeTD_PZ', 'CeTD_CH', 'CeTD_SS', 'CeTD_SA']
    head2 = ['CeTD_11', 'CeTD_12', 'CeTD_13', 'CeTD_21', 'CeTD_22', 'CeTD_23', 'CeTD_31', 'CeTD_32', 'CeTD_33']
    head3 = ['CeTD_0_p', 'CeTD_25_p', 'CeTD_50_p', 'CeTD_75_p', 'CeTD_100_p']
    head4 = ['HB', 'VW', 'PO', 'PZ', 'CH', 'SS', 'SA']
    for i in head1:
        for j in range(1, 4):
            ctd_title.append(i + str(j))
    for i in head2:
        for j in ['HB', 'VW', 'PO', 'PZ', 'CH', 'SS', 'SA']:
            ctd_title.append(i + '_' + str(j))
    for j in range(1, 4):
        for k in head4:
            for i in head3:
                ctd_title.append(i + '_' + k + str(j))
    attrs = [
        ['hydrophobicity', 'RKEDQN', 'GASTPHY', 'CLVIMFW'],
        ['normalized vander Waals volume', 'GASTPD', 'NVEQIL', 'MHKFRYW'],
        ['polarity', 'LIFWCMVY', 'PATGS', 'HQRKNED'],
        ['polarizability', 'GASDT', 'CPNVEQIL', 'KMHFRYW'],
        ['charge', 'KR', 'ANCQGHILMFPSTWYV', 'DE'],
        ['secondary structure', 'EALMQKRH', 'VIYCWFT', 'GNPSD'],
        ['solvent accessibility', 'ALFCGIVW', 'RKQEND', 'MSPTHY']
    ]
    std_list = [1, 2, 3]
    tr_list = [11, 12, 13, 21, 22, 23, 31, 32, 33]
    ctd_df = pd.DataFrame(columns=ctd_title)
    for seq in seqs:
        st = []
        tr1 = []
        cc1 = []
        for attr in attrs:
            st1 = []
            cc2 = []
            for s in list(seq):
                st1.extend([i for i in range(1, 4) if s in attr[i]])
            tr_st1 = [st1[i]*10+st1[i+1] for i in range(0, len(st1)-1)]
            for stdid in std_list:
                conter = Counter(st1)
                st.append(round((conter.get(stdid, 0) / len(st1)) * 100, 2))
                cc2.append([index for index, value in enumerate(st1) if value == stdid])
            for c2 in cc2:
                for e in range(0, 101, 25):
                    cc1.append(math.floor(e*(len(c2))/100))
            for trid in tr_list:
                conter = Counter(tr_st1)
                tr1.append(conter.get(trid, 0))
        comp = st + tr1 + cc1
        df = pd.DataFrame(columns=ctd_title, data=[comp])
        if ctd_df.empty is True:
            ctd_df = df
        else:
            ctd_df = ctd_df.append(df, ignore_index=True)
    ctd_df['seqs'] = seqs
    return ctd_df

def feature_gen(seqs):
    sf = ['ATC_N', 'CeTD_SS1', 'CeTD_VW2', 'BTC_S', 'CeTD_VW1', 'CeTD_PZ1', 'DDR_P', 'CeTD_SA3', 'DDR_G', 'CeTD_CH3',
          'CeTD_PO3', 'DDR_L', 'AAC_S', 'DDR_Y', 'DDR_Q', 'APAAC1_R', 'AAC_Y', 'DDR_V', 'DDR_R', 'AAC_K', 'DDR_A',
          'DDR_D', 'AAC_N', 'DDR_H', 'AAC_I', 'DDR_F', 'CeTD_32_HB', 'CeTD_31_PZ', 'DDR_C', 'DDR_M', 'CeTD_100_p_SS2',
          'AAC_E', 'CeTD_23_VW', 'AAC_C', 'DDR_E', 'DPC1_YP', 'DPC1_YA', 'DPC1_RP', 'DDR_W', 'DPC1_SH', 'DPC1_GF',
          'DPC1_RF', 'DPC1_PS', 'DPC1_KL', 'DPC1_IL', 'DPC1_FY', 'DPC1_RK', 'DPC1_ER', 'DPC1_DV', 'DPC1_VG', 'TPC_RQR',
          'DPC1_VQ', 'DPC1_RE', 'DPC1_ME', 'DPC1_PF', 'DPC1_RA', 'DPC1_AG', 'DPC1_KR', 'DPC1_FD', 'DPC1_PG', 'DPC1_HA',
          'DPC1_GL', 'DPC1_SF', 'DPC1_VS', 'DPC1_PW', 'DPC1_GM', 'DPC1_SI', 'DPC1_WK', 'DPC1_PP', 'DPC1_AY', 'DPC1_AR',
          'DPC1_VI', 'DPC1_GY', 'DPC1_PA', 'DPC1_SG', 'DPC1_FS', 'DPC1_LF', 'DPC1_YH', 'DPC1_FA', 'DPC1_WR']
    atc_df = atc(seqs)
    bond_df = bond(seqs)
    aac_df = aac_comp(seqs)
    dpc_df = dpc_comp(seqs)
    tpc_df = tpc_comp(seqs)
    apacc_1_df = apacc_1(seqs) ####
    apacc_df = apacc(apacc_1_df, aac_df)
    ddor_df = DDOR(seqs)
    ctd_df = ctd(seqs)
    df = pd.merge(atc_df, bond_df, on='seqs')
    df = pd.merge(df, aac_df, on='seqs')
    df = pd.merge(df, dpc_df, on='seqs')
    df = pd.merge(df, tpc_df, on='seqs')
    df = pd.merge(df, apacc_df, on='seqs')
    df = pd.merge(df, ddor_df, on='seqs')
    df = pd.merge(df, ctd_df, on='seqs')
    return df[sf]

def get_b3pred(seqs, df, logger, predictor_path):
    logger.info("begin run b3pred")
    df1 = feature_gen(seqs)
    model = predictor_path / 'm_bp3pred'
    clf = joblib.load(model)
    result = np.round(clf.predict_proba(df1)[:,-1], decimals=2)
    df1 = pd.DataFrame(columns=['peptides', 'prob_b3pred'])
    df1['peptides'] = seqs
    df1['prob_b3pred'] = result
    df['b3pred'] = df1
    logger.info(f'end for b3pred: \n {df1}')
    return None