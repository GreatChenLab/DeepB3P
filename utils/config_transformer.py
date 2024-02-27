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
    def __init__(self, d_model=64, d_ff=16, d_k=32, n_layers=1, n_heads=3, lr=0.0001, drop=0.3):
        super(Config, self).__init__()
        self.d_model = d_model  # embedding size
        self.d_ff = d_ff  # feedforeard dimension
        self.d_k = d_k  # dimension of K(=Q), V
        self.n_layers = n_layers  # number of encoder of decoder layer
        self.n_heads = n_heads
        self.drop = drop
        self.lr = lr
        self.checkpoint = self.base_dir / 'model/checkpoint'
        self.model_file = self.checkpoint / f'{self.d_model}d_{self.d_ff}ff_{self.d_k}k_{self.n_layers}nl_{self.n_heads}h_{self.lr}lr_{self.drop}p'

    def make_dir(self):
        if not os.path.exists(self.checkpoint):
            #logging.info(f'create dir for checkpoint {self.checkpoint}')
            os.makedirs(self.checkpoint)
            #logging.info(f'create dir for model file {self.model_file}')
            os.makedirs(self.model_file)
        if not os.path.exists(self.model_file):
            #logging.info(f'create dir for model file {self.model_file}')
            os.makedirs(self.model_file)
        if not self.reload and os.path.exists(self.checkpoint):
            #logging.info('remove dir for checkpoint')
            import shutil
            shutil.rmtree(self.checkpoint)
