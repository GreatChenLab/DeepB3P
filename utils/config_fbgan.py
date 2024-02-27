# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : config.py


import os
import warnings
from utils.utils import *
from utils.config import BaseConfig

warnings.filterwarnings('ignore')

class Config(BaseConfig):
    def __init__(self,
                 n_epochs=1000,
                 n_sequences=24,
                 hidden=128,
                 n_heads=4,
                 bs=8,
                 d_steps=5,
                 lr=0.0001,
                 cutoff=0.8,
                 is_blast=True,
                 is_predict=False,
                 load_old=True
                 ):
        super(Config, self).__init__()
        self.n_epochs = n_epochs
        self.n_sequences = n_sequences
        self.hidden = hidden
        self.n_heads = n_heads
        self.batch_size = bs
        self.d_steps = d_steps
        self.lr = lr
        self.cutoff = cutoff
        self.is_blast = is_blast
        self.is_predict = is_predict
        self.load_old = load_old
        self.predictor_path = self.base_dir / 'fbgan/predictor'
        self.blast_path = self.predictor_path / 'db'
        self.checkpoint = self.base_dir / 'fbgan/checkpoint'
        self.sample_dir = self.base_dir / 'fbgan/samples'
        self.out_dir = self.base_dir / 'fbgan/out'

    def make_dir(self):
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if not self.reload:
            #logging.info('remove dir for checkpoint')
            import shutil
            if os.path.exists(self.checkpoint):
                shutil.rmtree(self.checkpoint)
            if os.path.exists(self.sample_dir):
                shutil.rmtree(self.sample_dir)
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)