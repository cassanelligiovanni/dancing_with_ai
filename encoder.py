import numpy as np
import torch
import torch.nn as nn
from layers import MultiHeadAttention, PositionwiseFeedForward
from utils.utils import *

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):

        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)

        return enc_output


class Encoder(nn.Module):

    def __init__(
            self, max_seq_len=142, music_size=439,
            n_layers=2, n_head=8, d_k=64, d_v=64,
          d_model=240, d_inner=1024, dropout=0.1):

        super().__init__()

        self.d_model = d_model
        n_position = max_seq_len + 1

        self.src_emb = nn.Linear(music_size, d_model)

        self.position_enc = nn.Embedding.from_pretrained(
            positional_encoding_M(n_position, d_model, padding_idx=0),
            freeze=True)

        self.transformerLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_inner, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformerLayer, num_layers=n_layers)

        self.layer_stack = nn.ModuleList([
                    EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                    for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        enc_output = self.src_emb(src_seq)
        enc_output += self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output  = enc_layer(enc_output)

        # enc_output = self.transformer(enc_output)

        return enc_output,


