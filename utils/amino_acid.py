# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : amino_acid.py


from Bio.SeqIO import parse
import pandas as pd
import numpy as np
from torch.nn import functional as F

PAD = '0'
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

ID_TO_AMINO_ACID = {}
AMINO_ACID_TO_ID = {}
AMINO_ACIDS_TO_ONE_HOT = {}

AMINO_ACID_SIZE = len(AMINO_ACIDS)
ONE_HOT = np.eye(AMINO_ACID_SIZE+1, AMINO_ACID_SIZE, k=-1)

for index, value in enumerate(list(PAD + AMINO_ACIDS)):
    ID_TO_AMINO_ACID[index] = value
    AMINO_ACID_TO_ID[value] = index
    AMINO_ACIDS_TO_ONE_HOT[value] = ONE_HOT[index].tolist()

NON_STANDARD_AMINO_ACIDS = ['B', 'O', 'U', 'X', 'Z', 'J']


def from_amino_acid_to_id(data):
    """
    :param data: contains amino acid that need to be converted to ids
    :return: list of ids
    """
    return [AMINO_ACID_TO_ID[c] for c in data]

def from_id_to_amino_acid(data):
    """
    :param data: contains ids that need to be converted to amino acid
    :return: list of amino acid
    """
    return [ID_TO_AMINO_ACID[id] for id in data]

def fasta_to_pandas(path):
    """
    :param path: fasta file path
    :return:
    """
    records = parse(path, 'fasta')
    seqs = []
    for record in records:
        seqs.append(str(record.seq))

    return pd.DataFrame(columns=['peptide'], data=seqs)

def pandas_to_numpy(path, max_length=50, label=None):
    """
    :param path: csv file,title ['peptide', 'label']
    :return: features and labels
    """
    sequences, labels = [], []
    if not isinstance(path, pd.DataFrame):
        df = pd.read_csv(path)
    else:
        df = path
    for index, row in df.iterrows():
        seq = str(row['peptide'])[:max_length]
        if label is None:
            label = int(row['label'])
        seq = seq.ljust(max_length, '0')
        np_seq = np.asarray(from_amino_acid_to_id(seq))
        sequences.append(np_seq)
        labels.append(np.asarray(label))
    return np.stack(sequences, axis=0).astype(np.int64), np.stack(labels, axis=0).astype(np.int64)


def fasta_to_numpy(path, max_length=50, label=1):
    """
    :param path: the fasta file
    :param length: max length for the sequence
    :return: numpy array of sequences and labels
    """
    sequences = []
    labels = []
    records = parse(path, 'fasta')
    for record in records:
        seq = str(record.seq)[:max_length]
        seq = seq.ljust(max_length, '0')
        np_seq = np.asarray(from_amino_acid_to_id(seq))
        sequences.append(np_seq)
        labels.append(np.asarray(label))
    if sequences:
        return np.stack(sequences, axis=0).astype(np.int64), np.stack(labels, axis=0).astype(np.int64)
    else:
        return None, None

def remove_pad_from_str_seq(seqs, pad):
    """
    :param seqs: seqs contains pad
    :param pad:
    :return:
    """
    seqs = ''.join(seqs).rstrip(pad)
    if pad in seqs or len(seqs) < 5:
        return None
    return seqs

def write_fasta(seqs, path):
    f = open(path, 'w')
    for index, value in enumerate(seqs):
        f.write('>'+str(index)+'\n')
        f.write(value+'\n')
    f.close()