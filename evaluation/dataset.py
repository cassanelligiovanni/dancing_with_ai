import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn.functional as F
from classifier import Classifier
import time

def prepare_dataloader(dance_data, labels, batch_size):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(dance_data, labels),
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=paired_collate_fn,
        # pin_memory=True
    )

    return data_loader


class DanceDataset(Dataset):
    def __init__(self, dances, labels=None):
        if labels is not None:
            assert (len(labels) == len(dances)), \
                'the number of dances should be equal to the number of labels'
        self.dances = dances
        self.labels = labels

    def __len__(self):
        return len(self.dances)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.dances[index], self.labels[index]
        else:
            return self.dances[index]

def load_data(data_dir, interval):
    dance_data, labels = [], []
    fnames = sorted(os.listdir(data_dir))

    # fnames = fnames[:1000]  # For debug

    for fname in fnames:
        if fname != ".DS_Store":

            path = os.path.join(data_dir, fname)
            label = -1

            if "BR" in fname:
                label = 0
            elif "HO" in fname:
                label = 1
            elif "MH" in fname:
                label = 2
            elif "KR" in fname:
                label = 3
            elif "LH" in fname:
                label = 4
            elif "LO" in fname:
                label = 5
            elif "PO" in fname:
                label = 6
            elif "WA" in fname:
                label = 7
            elif "JB" in fname:
                label = 8
            elif "JS" in fname:
                label = 9




            with open(path) as f:
                sample_dict = json.loads(f.read())
                np_dance = np.array(sample_dict['motion'])
                # Only use 25 keypoints skeleton (basic bone) for 2D
                np_dance = np_dance[:, :51]

                dance_data.append(np_dance)
                labels.append(label)

    return dance_data, labels


