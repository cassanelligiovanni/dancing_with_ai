import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import *


class DataSet(Dataset):
    def __init__(self, musics, dances):
        self.musics = musics
        self.dances = dances

    def __len__(self):
        return len(self.musics)

    def __getitem__(self, index):
        if self.dances is not None:
            return self.musics[index], self.dances[index]
        else:
            return self.musics[index]


def prepare_dataloader(music_data, dance_data, batch_size=32):

   data_loader = DataLoader(
        DataSet(music_data, dance_data),
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=generate_positions,
    )

   return data_loader

def load_data(data_dir):
   music_data, dance_data = [], []
   fnames = os.listdir(data_dir)
   # fnames = fnames[:10]  # For debug

   for i, fname in enumerate(fnames):
      path = os.path.join(data_dir, fname)
      if fname != ".DS_Store":
          with open(path) as f:

             sample = json.loads(f.read())
             music = np.array(sample['music'])
             dance = np.array(sample['motion'])

             root = dance[:, 3*11:3*12]
             dance = dance - np.tile(root, (1, 17))
             dance[:, 3*11:3*12] = root

             music_data.append(music)
             dance_data.append(dance)

             sys.stderr.write('\rLoading Dataset %d / %d' % (i + 1, len(fnames)))

   return music_data, dance_data

