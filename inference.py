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

from utils.utils import *
from model import *
from dataset import *
from interpolate_to_60fps import *

np.random.seed(0)
torch.manual_seed(0)

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "/temp-motion", "")

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('d_model',"240", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('n_layers',"2", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('n_heads',"8", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('inner_d',"1024", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('learning_rate',"0.0001", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('lambda_v',"0.01", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('dropout',"0.1", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('max_epochs',"5000", "path to normalised 3d keypoints + extension")


def main(_):

    data_dir = FLAGS.data_dir
    test_dir = os.path.join(data_dir, "dataset/test/")
    predicted_dir = os.path.join(data_dir, "dataset/predicted/")
    model_parameters =FLAGS.model

    # Model Parameters
    d_model = int(FLAGS.d_model)
    n_layers = int(FLAGS.n_layers)
    n_heads = int(FLAGS.n_heads)
    inner_d = int(FLAGS.inner_d)
    MUSIC_SIZE = 439
    DANCE_SIZE = 51
    D_K, D_V = 64, 64

    # Training  Hyper-parameters
    learningRate = float(FLAGS.learning_rate)
    lambda_v = float(FLAGS.lambda_v)
    maxEpochs = int(FLAGS.max_epochs)
    batch_size = 16
    DROPOUT = float(FLAGS.dropout)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(max_seq_len=142,
                      music_size=MUSIC_SIZE,
                      n_layers=n_layers,
                      n_head=n_heads,
                      d_k=D_K,
                      d_v=D_V,
                      d_model=d_model,
                      d_inner=inner_d,
                      dropout=DROPOUT)

    decoder = Decoder(motion_size=DANCE_SIZE,
                      d_emb=d_model,
                      hidden_size=inner_d,
                      dropout=DROPOUT)


    model = Model(encoder, decoder,
                  condition_step=10,
                  lambda_v=lambda_v,
                  device=device)




    pretrained_dict = torch.load(model_parameters, map_location='cpu')
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}

    model.load_state_dict(pretrained_dict)
    model.eval()

    with torch.no_grad():

      test_audios = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith(".json")]

      for audio in test_audios :

          with open(audio) as f :

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

          save_obj(interpolated, predicted_dir, name)

if __name__ == '__main__':
  app.run(main)
