# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : classifier.py


import torch.nn.functional as F
from model.transLayers import *

class Classifier(nn.Module):
    def __init__(self, seq_len, vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device):
        super(Classifier, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device).to(device)
        self.decoder = Decoder(vocab_size, seq_len, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device).to(device)
        self.projection = nn.Sequential(
            nn.Linear(d_model * seq_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        enc_outputs, self.enc_att = self.encoder(x) #enc_outputs: [batch_size, seq_len, d_model], enc_att: [batch_size, n_layers, n_heads, src_len, src_len]
        dec_outputs, self.dec_att, self.dec_enc_att = self.decoder(x, enc_outputs) #dec_outputs: [batch_size, seq_len, d_model], enc_att: [batch_size, n_layers, n_heads, src_len, src_len]
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)
        logists = self.projection(dec_outputs)
        return F.softmax(logists)

    def get_atts(self):
        return self.enc_att, self.dec_att, self.dec_enc_att

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
