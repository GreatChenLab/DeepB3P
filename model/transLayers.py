# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : transLayers.py

import torch
import torch.nn as nn
import math
import numpy as np

def get_attn_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [bs, seq_len]
    :param seq_k:
    :return:
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, len_q, d_k]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v(=len_k), d_v]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]
        :return:
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, device):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.device = device
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch_size, n_heads, len_q, d_model]
        :param input_K: [batch_size, n_heads, len_k, d_model]
        :param input_V: [batch_size, n_heads, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, device):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [bs, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual) # [bs, seq_len, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        """
        :param x: [seq_len, bs, d_model]
        :return:
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, device)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [bs, seq_len, d_model]
        :param enc_self_attn_mask: [bs, seq_len, src_len]
        :return: enc_outputs: [batch_size, seq_len, d_model], attn: [batch_size, n_heads, seq_len, seq_len]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, device, drop)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff, device) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size, seq_len]
        :return: enc_outputs: [batch_size, seq_len, d_model], enc_self_attns: [batch_size, n_heads, seq_len, seq_len]
        """
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, seq_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, seq_len, seq_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, device):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, device)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, device)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask):
        """
        :param dec_inputs: [batch_size, seq_len, d_model]
        :param enc_outputs: [batch_size, seq_len, d_model]
        :param dec_self_attn_mask: [batch_size, seq_len, seq_len]
        :return: dec_outputs:[batch_size, seq_len, d_model], dec_self_attn, dec_enc_attn:[batch_size, n_heads, seq_len, seq_len]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, d_k, d_v, d_ff, n_layers, drop, device):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, device, drop)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_ff, device) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        """
        :param dec_inputs: [batch_size, seq_len]
        :param enc_outputs: [batch_size, seq_len, d_model]
        :return:
        """
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(self.device)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(self.device)
        dec_self_attns = []
        dec_enc_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
