# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : classifier.py


import math
import torch.nn.functional as F

from model.layers import *

class Classifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_size, hidden, n_heads, max_pool, dropout, device):
        super(Classifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0) # bs, L, e_s
        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=hidden, kernel_size=2, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=embedding_size, out_channels=hidden, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=embedding_size, out_channels=hidden, kernel_size=4, stride=1, padding='same')
        self.conv4 = nn.Conv1d(in_channels=embedding_size, out_channels=hidden, kernel_size=5, stride=1, padding='same')
        self.mha = nn.MultiheadAttention(embed_dim=hidden, num_heads=n_heads, batch_first=True, device=device) #bs, L, E_q
        self.bigru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True, bidirectional=True)#bs, L, E_q

        self.fc1 = nn.Linear(seq_len * hidden * 2, 1000)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)

    def multi_scale_cnn(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        x = x1 + x2 + x3 + x4
        x = self.dropout(x) #bs, hidden, seq_len
        return x

    def forward(self, x):
        """
        :param x: bs, seq_len
        :return:
        """
        embed_output = self.embed(x) # bs, seq_len, embed_size
        cnn_input = embed_output.transpose(1, 2)
        cnn_output = self.multi_scale_cnn(cnn_input) #bs, hidden, seq_len
        out = cnn_output.transpose(1, 2) #bs, seq_len, hidden
        out, att = self.mha(out, out, out) # (bs, seq_len, hidden), (bs, seq_len, seq_len)
        out, _ = self.bigru(out) #bs, seq_len, hidden*2
        out = out.contiguous().view(out.shape[0], -1) #bs, seq_len * hidden * 2
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        logists = F.relu(self.fc5(out))
        return F.softmax(logists)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)