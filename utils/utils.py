import torch
import torch.nn as nn
import numpy as np

from absl import app
from absl import flags
import pickle

import json

def save_json(file, path) :

    with open(path, 'w') as f:

        json.dump(file, f)





def loss_function(input,target):

    criterion = nn.MSELoss()
    loss = criterion(input, target)

    return loss


def convert_back_to_3d(batch, max_val, mean_pose) :

    max_val_torch = torch.from_numpy(max_val)
    mean_pose_torch = torch.from_numpy(mean_pose)

    # print("####### batch size ######", batch.shape)

    # print("####### max_val_torch size ######", max_val_torch.shape)

    reconstructed = torch.mul(batch, max_val_torch)  + 1e-15

    # print("####### pre-mean ######", reconstructed[0, 0])
    reconstructed = reconstructed + mean_pose_torch
    # print("####### mean ######", mean_pose_torch[0])

    # print("####### port-mean ######", reconstructed[0, 0])
    return reconstructed

def add_noise(batch, sigma=0, variance=0.05):

    eps = 1e-15

    mean = torch.zeros_like(batch)
    stddev = np.multiply(sigma, variance)+eps

    noise = torch.normal(mean, stddev)
    # print("#### NOISE SHAPE ###", noise.shape)
    batch = batch  + noise
    return batch


def save_obj(obj, directory, name ):
    with open(directory + name +  '.pkl', 'wb') as f:
        pickle.dump(obj, f, 4)


