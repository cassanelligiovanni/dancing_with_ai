import torch
import torch.nn as nn
import numpy as np

from absl import app
from absl import flags
import pickle
import wandb
import json

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)


def save_json(file, path) :
    with open(path, 'w') as f:
        json.dump(file, f)

def loss_function(input,target):
    criterion = nn.MSELoss()
    loss = criterion(input, target)
    return loss

def save_obj(obj, directory, name ):
    with open(directory + name +  '.pkl', 'wb') as f:
        pickle.dump(obj, f, 4)

def generate_positions(insts):
    src_seq, tgt_seq = list(zip(*insts))
    src_pos = np.array([
        [pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in src_seq])

    src_seq = torch.FloatTensor(src_seq)
    src_pos = torch.LongTensor(src_pos)
    tgt_seq = torch.FloatTensor(tgt_seq)

    return src_seq, src_pos, tgt_seq


def positional_encoding_M(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)



