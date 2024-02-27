# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : blast.py

import subprocess
import os
from utils.amino_acid import write_fasta

def get_local_blast_results(data_dir, db_path, seqs):
    query_path = os.path.join(data_dir, 'blast_query.fasta')
    write_fasta(seqs, query_path)

    blastp = subprocess.Popen(
        ['blastp', '-db', str(db_path), '-max_target_seqs', '1', '-outfmt', '10', '-matrix', 'BLOSUM45',
         '-task', 'blastp-short', '-query', str(query_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    results, err = blastp.communicate()
    if err.decode():
        return None
    return parse_blast_results(results.decode(), seqs)


def parse_blast_results(results, seqs):
    parsed = {}
    for line in results.split(os.linesep):
        line = line.strip()
        parts = line.split(',')
        if len(parts) != 12:
            continue
        parsed[seqs[int(parts[0])]] = float(parts[-1])

    return parsed

if __name__ == '__main__':
    seqs = ['VLGGGSALLRSIPA', 'FCIGRL', 'CHAIYPRH']
    data_path = BASE_DIR / 'fbgan/predictor/db'
    pos_db = data_path / 'pos'
    neg_db = data_path / 'neg'
    pos = get_local_blast_results(data_path, pos_db, seqs)
    print('blast pos\n {0}'.format(pos))
    neg = get_local_blast_results(data_path, neg_db, seqs)
    print('blast neg\n {0}'.format(neg))

