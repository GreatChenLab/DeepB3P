# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : fbgan_model.py


from fbgan.fbgan import *
from utils.config_fbgan import Config


def main():
    import gc
    gc.collect()
    params = Config()
    params.make_dir()
    logfile = params.out_dir / 'fbgan.log'
    logger = params.set_logging(logfile)
    set_seed(2022, logger)
    pos_data = './data/pos.fasta'
    features, labels = fasta_to_numpy(pos_data)
    logger.info('read seqs: {0}'.format(len(labels)))
    logger.info(f'n_epochs:\t{params.n_epochs}\nbs:\t{params.bs}\ncutoff:\t{params.cutoff}\nload_old:\t{params.load_old}')

    model = FBGANBBB(params, logger)
    logger.info('load old model is {0}'.format(params.load_old))
    model.train_model(x_data=features, y_data=labels)
if __name__ == '__main__':
    main()