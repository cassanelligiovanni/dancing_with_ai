import numpy as np
import torch
import torch.nn as nn
from utils.pose import INITIAL_POSE
from layers import MultiHeadAttention, PositionwiseFeedForward
from utils.utils import *


class Decoder(nn.Module):
    def __init__(self, motion_size=51, d_emb=51, hidden_size=1024,
                 dropout=0.1):
        super().__init__()

        self.motion_size = motion_size
        self.d_emb = d_emb
        self.hidden_size = hidden_size

        self.tgt_emb = nn.Linear(motion_size, d_emb)
        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTMCell(d_emb, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)

    def initialise(self, batch_size, device):

        c0 = torch.randn(batch_size, self.hidden_size).to(device)
        c1 = torch.randn(batch_size, self.hidden_size).to(device)
        c2 = torch.randn(batch_size, self.hidden_size).to(device)
        h0 = torch.randn(batch_size, self.hidden_size).to(device)
        h1 = torch.randn(batch_size, self.hidden_size).to(device)
        h2 = torch.randn(batch_size, self.hidden_size).to(device)

        vec_h = [h0, h1, h2]
        vec_c = [c0, c1, c2]

        initial_pose = INITIAL_POSE

        initial_pose = np.tile(initial_pose, (batch_size, 1))
        root = initial_pose[:, 3*11:3*12]
        translated = initial_pose - np.tile(root, (1, 17))
        translated[:,  3*11:3*12] = root
        out_frame = torch.from_numpy(translated).float().to(device)
        out_seq = torch.FloatTensor(batch_size, 1).to(device)

        return (vec_h, vec_c), out_frame, out_seq

    def forward(self, in_frame, vec_h, vec_c):

        in_frame = self.tgt_emb(in_frame)
        in_frame = self.dropout(in_frame)

        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]
        return vec_h2, vec_h_new, vec_c_new


