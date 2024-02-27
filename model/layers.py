# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : layers.py

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, n_hiddens, dropout, max_len=1000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros(1, max_len, n_hiddens).to(device)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, n_hiddens, 2, dtype=torch.float32) / n_hiddens
        )

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :]
        return self.dropout(x)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, ffn_input, ffn_hidden, ffn_output):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ffn_input, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, ffn_output)
        )
    def forward(self, x):
        return self.fc(x)


class AttentionLayer(nn.Module):
    def __init__(self, embedding_size, n_heads, dropout, device):
        super(AttentionLayer, self).__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.att = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=n_heads, batch_first=True, device=device)
        self.layer_norm = nn.LayerNorm(normalized_shape=[50, embedding_size])

    def forward(self, x, y=None):
        out, weight = self.att(x, x, x)
        multi_encode = self.layer_norm(self.dropout(out) + x)
        return multi_encode


class FFN_encode(nn.Module):
    def __init__(self, dropout, shape):
        super(FFN_encode, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=[1, shape])
        self.ffn = PoswiseFeedForwardNet(ffn_input=shape, ffn_hidden=(2*shape), ffn_output=shape)

    def forward(self, x):
        encode_output = self.layer_norm(x + self.dropout(self.ffn(x)))
        return encode_output
