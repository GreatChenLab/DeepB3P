# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : models.py

import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=5, padding=2),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)


class Generator(nn.Module):
    def __init__(self, n_chars, seq_len, bs, hidden):
        super(Generator, self).__init__()
        self.bs = bs
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.hidden = hidden
        self.fc1 = nn.Linear(128, hidden * seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            #ResBlock(hidden),
            #ResBlock(hidden),
            #ResBlock(hidden),
            #ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(in_channels=hidden, out_channels=n_chars, kernel_size=1)

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len)
        output = self.block(output)
        output = self.conv1(output) #[bs, n_chars, seq_len]
        output = output.transpose(1, 2).contiguous()
        output = output.view(self.bs * self.seq_len, -1)
        output = F.gumbel_softmax(output, 0.5)
        output = output.view(self.bs, self.seq_len, self.n_chars)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_chars, seq_len, hidden, n_head):
        super(Discriminator, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlock(hidden),
            #ResBlock(hidden),
            #ResBlock(hidden),
            #ResBlock(hidden),
            #ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len * hidden, 1)

    def forward(self, input):
        """
        :param input: [bs, seq_len, n_chars]
        :return:
        """
        output = input.transpose(1, 2) # [bs, n_chars, seq_len]
        output = output.contiguous().float()  # [bs, n_chars, seq_len]
        output = self.conv1d(output)
        output = self.block(output) #[bs, hidden, seq_len]
        output = output.contiguous().view(-1, self.seq_len * self.hidden)
        output = self.linear(output)
        return output