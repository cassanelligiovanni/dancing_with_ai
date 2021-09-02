import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(data_dir):
   music_data, dance_data = [], []
   # fnames = sorted(os.listdir(data_dir))
   # fnames = fnames[:10]  # For debug
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

def prepare_dataloader(music_data, dance_data, batch_size=32):

   data_loader = DataLoader(
        DataSet(music_data, dance_data),
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=paired_collate_fn,
        # pin_memory=True
    )

   return data_loader

def paired_collate_fn(insts):
    src_seq, tgt_seq = list(zip(*insts))
    src_pos = np.array([
        [pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in src_seq])

    src_seq = torch.FloatTensor(src_seq)
    src_pos = torch.LongTensor(src_pos)
    tgt_seq = torch.FloatTensor(tgt_seq)

    return src_seq, src_pos, tgt_seq




class DataSet(Dataset):
    def __init__(self, musics, dances):
        if dances is not None:
            assert (len(musics) == len(dances)), \
                'the number of dances should be equal to the number of musics'
        self.musics = musics
        self.dances = dances

    def __len__(self):
        return len(self.musics)

    def __getitem__(self, index):
        if self.dances is not None:
            return self.musics[index], self.dances[index]
        else:
            return self.musics[index]

