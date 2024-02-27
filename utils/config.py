# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : config.py


import torch
import logging
from pathlib import Path

class BaseConfig(object):
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.seq_len = 50
        self.vocab_size = 21  # 20 aa and 1 pad
        self.drop = 0.3
        self.seed = 2023
        self.bs = 16
        self.n_epochs = 200
        self.lr = 0.002
        self.kFold = 5
        self.earlyStop = 5
        self.reload = True
        self.base_dir =  Path(__file__).resolve().parent.parent
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    def make_dir(self):
        pass

    def set_bacsic_logging(self, file):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            filename=file,
            filemode='w'
        )

    def set_logging(self, file):
        file = logging.FileHandler(filename=file, mode='w', encoding='utf-8')
        format = logging.Formatter(fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')
        file.setFormatter(format)
        logger = logging.Logger(name='deepb3p.log', level=logging.INFO)
        logger.addHandler(file)
        return logger