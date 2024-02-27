# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/11/15
# email  : tangqiang.0701@gmail.com
# file   : predict_user.py

from model.deepb3p import DeepB3P
from utils.utils import *
from utils.config_transformer import Config
from utils.amino_acid import *
from tqdm import tqdm
from Bio.SeqIO import parse


def test(model, data_loader, device):
    y_true_list, y_prob_list, loss_list = [], [], []
    with torch.no_grad():
        model.model.eval()
        for feats, labels in tqdm(data_loader, mininterval=1, desc='Testing Processing', leave=False):
            feats, labels = feats.to(device), labels.to(device)
            outputs = model.model(feats)
            y_train = labels.cpu().detach().numpy()
            y_prob = outputs[:, 1].cpu().detach().numpy()
            y_true_list.extend(y_train)
            y_prob_list.extend(y_prob)
        y_pred_list = transfer(y_prob_list, 0.5)
        return y_true_list, y_pred_list, y_prob_list

def get_models(params, logger):
    model_list = []
    for i in range(1, params.kFold+1):
        model_file = params.model_file / f'deepb3p_{i}.pth'
        model = DeepB3P(params, logger)
        model.model.reset_parameters()
        model.load_model(model_file)
        model_list.append(model)
    return model_list


def cv_predict(params, dataloader, logger):
    model_list = get_models(params, logger)
    y_prob_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
    y_pred_df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
    for idx, model in enumerate(model_list):
        y_true_list, y_pred_list, y_prob_list = test(model, dataloader, params.device)
        y_prob_df[idx + 1] = y_prob_list
        y_pred_df[idx + 1] = y_pred_list
    return y_pred_df, y_prob_df

def get_seqs(file):
    seqs = []
    records = parse(file, 'fasta')
    for one in records:
        seqs.append(str(one.seq))
    return seqs

def predict(file):
    params = Config(
        d_model=512,
        d_ff=16,
        d_k=32,
        n_layers=1,
        n_heads=2,
        lr=0.0001,
        drop=0.1
    )
    params.make_dir()
    logfile = params.model_file / 'deepb3p.log'
    logger = params.set_logging(logfile)
    # if file.endswith('csv'):
    #     feas, label = pandas_to_numpy(file, label=1)
    feas, label = fasta_to_numpy(file, label=1)

    data_len = len(feas)
    test_dataset = SeqDataset(feas, label)
    dataloader = DataLoader(test_dataset, batch_size=data_len)
    y_pred_df, y_prob_df = cv_predict(params, dataloader, logger)
    avg_prob = np.round(y_prob_df.mean(axis=1), 4)
    res = pd.DataFrame(avg_prob, columns=['prob'])
    seqs = get_seqs(file)
    res.insert(0, 'peptide', value=seqs)
    res.to_csv('prob.txt', index=False)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage:\npython predict_user.py fasta_file")
        exit(0)
    file = sys.argv[1]
    predict(file)
    print('predict ok')