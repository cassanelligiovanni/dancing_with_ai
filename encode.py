import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
import torch.nn as nn
from torch.nn import Module
import torchvision
from torch.nn.utils.rnn import pack_sequence

from utils import *
from DAE import *

def encode_motion(to_encode_path, model_parameters, encoded_dir, encoded_name):

    motion = np.load(to_encode_path, allow_pickle=True)
    eps = 1e-8

    encoder = Encoder()

    parameters = torch.load(model_parameters)
    encoder.load_state_dict(parameters['encoder'])
    encoder.eval()

    max_value = parameters['max_value']
    mean_pose = parameters['mean_pose']

    motion_flattened = motion.reshape(motion.shape[0], -1)
    motion_centered = motion_flattened - mean_pose[np.newaxis, :]
    motion_normalised = np.divide(motion_centered, max_value[np.newaxis, :] + eps)
    motion_torch = torch.from_numpy(motion_normalised).float()

    output = encoder.forward(motion_torch)
    save_obj(output, encoded_dir, encoded_name)


