import numpy as np
import torch
import torch.nn as nn
from utils.pose import INITIAL_POSE
from layers import MultiHeadAttention, PositionwiseFeedForward
from encoder import Encoder
from decoder import Decoder
from utils.utils import *


class Model(nn.Module):
    def __init__(self, encoder, decoder, condition_step=10,  lambda_v=0.01, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(decoder.hidden_size + encoder.d_model, decoder.motion_size)

        self.condition_step = condition_step
        self.lambda_v = lambda_v
        self.device = device

    def initialise_decoder(self, batch_size):
        return self.decoder.initialise(batch_size, self.device)

    def forward(self, src_seq, src_pos, tgt_seq, hidden, dec_output, out_seq, epoch_i):
        batch_size, seq_len, _ = tgt_seq.size()
        vec_h, vec_c = hidden

        enc_outputs, *_ = self.encoder(src_seq, src_pos)

        groundtruth_mask = torch.ones(seq_len, self.condition_step)
        prediction_mask = torch.zeros(seq_len, int(epoch_i * self.lambda_v))
        mask = torch.cat([prediction_mask, groundtruth_mask], 1).view(-1)[:seq_len]

        preds = []
        for i in range(seq_len):
            dec_input = tgt_seq[:, i] if mask[i] == 1 else dec_output.detach()
            dec_output, vec_h, vec_c = self.decoder(dec_input, vec_h, vec_c)
            dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
            dec_output = self.linear(dec_output)
            preds.append(dec_output)

        outputs = [z.unsqueeze(1) for z in preds]
        outputs = torch.cat(outputs, dim=1)
        return outputs
