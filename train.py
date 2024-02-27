# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : train.py

from model.deepb3p import DeepB3P
from utils.config_transformer import *
from utils.amino_acid import *
from validation import predict

train_pos_org = './data/train_pos_org.fasta'
train_pos_pseudo = './data/df_all_final_1000.csv'
train_neg_org = './data/neg_train.fasta'
pos_test = './data/pos_test.fasta'
neg_test = './data/neg_test.fasta'

train_pos_pseudo = pd.read_csv(train_pos_pseudo)
train_pos_org = fasta_to_pandas(train_pos_org)
pos_all = pd.concat([train_pos_org, train_pos_pseudo])


pos_train_fea, pos_train_label = pandas_to_numpy(pos_all, label=1)
neg_train_fea, neg_train_label = fasta_to_numpy(train_neg_org, label=0)
#logging.info('read pos train seqs: {0}'.format(len(pos_train_fea)))
#logging.info('read neg train seqs: {0}'.format(len(neg_train_fea)))

pos_test_fea, pos_test_label = fasta_to_numpy(pos_test)
neg_test_fea, neg_test_label = fasta_to_numpy(neg_test, label=0)
#logging.info('read pos test seqs: {0}'.format(len(pos_test_fea)))
#logging.info('read neg test seqs: {0}'.format(len(neg_test_fea)))

train_feas = np.concatenate((pos_train_fea, neg_train_fea), axis=0)
train_labels = np.concatenate((pos_train_label, neg_train_label), axis=0)


train_dataset = SeqDataset(train_feas, train_labels)

# d_model=64, d_ff=16, d_k=32, n_layers=1, n_heads=2, lr=0.0001, drop=0.3
d_model = 512
d_ff = 16
d_k = 32
n_layers = 1
n_heads = 2
lr = 0.0001
drop = 0.1
results = open('res.txt', 'w')
results.write('lr\tdrop\tn_head\tn_layers\td_k\td_ff\td_model\tauc\tsn\tsp\tacc\tmcc\n')
results.close()
params = Config(
    d_model=d_model,
    d_ff=d_ff,
    d_k=d_k,
    n_layers=n_layers,
    n_heads=n_heads,
    lr=lr,
    drop=drop
)
params.make_dir()
logfile = params.model_file / 'deepb3p.log'
logger = params.set_logging(logfile)
set_seed(2023, logger)
logger.info('read pos train seqs: {0}'.format(len(pos_train_fea)))
logger.info('read neg train seqs: {0}'.format(len(neg_train_fea)))
model = DeepB3P(params, logger)
model.cv_train(train_dataset, kFlod=params.kFold, earlyStop=params.earlyStop)
auc, sn, sp, acc, mcc = predict(params, logger)
with open('res.txt','a') as results:
    results.write(str(lr)+'\t'+str(drop)+'\t'+str(n_heads)+'\t'+
              str(n_layers)+'\t'+str(d_k)+'\t'+str(d_ff)+'\t'
              +str(d_model)+'\t')
    results.write(str(auc)+'\t'+str(sn)+'\t'+str(sp)+'\t'+str(acc)+'\t'+str(mcc)+'\n')
