import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
import torch.nn as nn
from torch.nn import Module
import torchvision
from torch.nn.utils.rnn import pack_sequence

from absl import app
from absl import flags

<<<<<<< HEAD
from interpolate_to_60fps import *
=======
>>>>>>> origin/main
from utils.utils import *
from model import *
from dataset import *

<<<<<<< HEAD
# np.random.seed(0)
# torch.manual_seed(0)
=======
np.random.seed(0)
torch.manual_seed(0)
>>>>>>> origin/main

FLAGS = flags.FLAGS
flags.DEFINE_string("motion", "/temp-motion", "")
flags.DEFINE_string("model", "/temp-motion", "")

def main(_):
    motion = FLAGS.motion
    model_parameters =FLAGS.model


    INPUT_SIZE = 439

    D_POSE_VEC = 51

    D_MODEL = 240
    N_LAYERS = 2
    N_HEAD = 8
    D_K, D_V = 64, 64
    D_INNER = 1024
    DROPOUT = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(max_seq_len=142,
                      music_size=INPUT_SIZE,
                      n_layers=N_LAYERS,
                      n_head=N_HEAD,
                      d_k=D_K,
                      d_v=D_V,
                      d_model=D_MODEL,
                      d_inner=D_INNER,
                      dropout=DROPOUT)

    decoder = Decoder(motion_size=D_POSE_VEC,
                      hidden_size=D_INNER,
                      d_emb=240,
                      dropout=DROPOUT)


    model = Model(encoder, decoder,
                  condition_step=10,
                  lambda_v=0.01,
                  device=device)

    pretrained_dict = torch.load(model_parameters, map_location='cpu')
    # pretrained_dict = torch.load(model_parameters, map_location='cpu')['twenty_step_model']

    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}

    model.load_state_dict(pretrained_dict)
    model.eval()

    with torch.no_grad():


       with open(motion) as f :

          file = json.loads(f.read())
          name = file["id"]
          music = np.array(file["music"])
          # audio_name ="_".join( name.split("_")[:-1])
          # music= np.load(audio_dir+ "/" + audio_name+".pkl", allow_pickle=True)
          music_tensor = torch.FloatTensor(music[None, :, :])

          pos =torch.LongTensor(np.arange(1, 143, 1))

          b, music_length, _ = music_tensor.size()
          # bsz, tgt_seq_len, dim = tgt_seq.size()
          tgt_seq_len = 1
          generated_frames_num = music_length - tgt_seq_len

          hidden, dec_output, out_seq = model.initialise_decoder(b)
          a, c = hidden

          enc_outputs, *_ = model.encoder(music_tensor, pos)

          preds = []
          for i in range(tgt_seq_len):
              # dec_input = tgt_seq[:, i]
              dec_input = dec_output
              dec_output, a, c = model.decoder(dec_input, a, c)
              dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
              dec_output = model.linear(dec_output)
              preds.append(dec_output)

          for i in range(generated_frames_num):
              dec_input = dec_output
              dec_output, a, c = model.decoder(dec_input, a, c)
              dec_output = torch.cat([dec_output, enc_outputs[:, i + tgt_seq_len]], 1)
              dec_output = model.linear(dec_output)
              preds.append(dec_output)


       final = np.array([np.array(x) for x in  preds])
       final = final.reshape(final.shape[0], 51)
       root = final[:,3*11:3*12]
       final = final + np.tile(root,(1,17))
       final[:,3*11:3*12] = root
       final = final.reshape(final.shape[0], 17, 3)

       interpolated = np.array(interpolate(final))

       save_obj(interpolated, "../gMH/predicted/",name )

if __name__ == '__main__':
  app.run(main)
