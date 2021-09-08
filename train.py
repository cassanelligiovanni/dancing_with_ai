import os

from absl import app
from absl import flags

import pickle
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

from utils.utils import *
from model import *
from dataset import *

np.random.seed(0)
torch.manual_seed(0)

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('audio_input_size',"439", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('motion_input_size',"51", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('d_model',"300", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('n_layers',"2", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('n_heads',"8", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('inner_d',"1024", "path to normalised 3d keypoints + extension")

def main(_):

    data_dir = FLAGS.data_dir

    audio_input_size = int(FLAGS.audio_input_size)
    motion_input_size = int(FLAGS.motion_input_size)
    d_model = int(FLAGS.d_model)
    n_layers = int(FLAGS.n_layers)
    n_heads = int(FLAGS.n_heads)
    inner_d = int(FLAGS.inner_d)


    D_K, D_V = 64, 64
    DROPOUT = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    encoder = Encoder(max_seq_len=2878,
                      input_size=audio_input_size,
                      d_word_vec=d_model,
                      n_layers=n_layers,
                      n_head=n_heads,
                      d_k=D_K,
                      d_v=D_V,
                      d_model=d_model,
                      d_inner=inner_d,
                      dropout=DROPOUT)

    decoder = Decoder(input_size=motion_input_size,
                      d_word_vec=motion_input_size,
                      hidden_size=inner_d,
                      encoder_d_model=d_model,
                      dropout=DROPOUT)


    model = Model(encoder, decoder,
                  condition_step=10,
                  sliding_windown_size=426,
                  lambda_v=0.01,
                  device=device)


    learningRate = 0.0001
    maxEpochs = 10000
    batch_size = 16

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    train_dir = data_dir + "dataset/train"

    music_data, dance_data = load_data(train_dir)
    loader = prepare_dataloader(music_data, dance_data, batch_size)

    optimizer = optim.Adam(filter(
        lambda x: x.requires_grad, model.parameters()), lr=learningRate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    criterion = nn.L1Loss()
    updates = 0

    # Set random seed
    random.seed(100)
    torch.manual_seed(200)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed(200)

    model = nn.DataParallel(model).to(device) if torch.cuda.is_available() else model.to(device)

    print(" ______________ ______")
    print("|     Epoch    | RMSE |")
    print("|------------  |------|")
    for epoch in range(maxEpochs):

        calculate_loss = False
        current_loss = 0

        model.train()

        for i, batch in enumerate(loader):

            music_seed, pos, dance_seed = map(lambda x: x.to(device), batch)
            target = dance_seed[:, 1:]
            music_seed = music_seed[:, :-1]
            pos = pos[:, :-1]
            dance_seed = dance_seed[:, :-1]

            if torch.cuda.is_available() :
                hidden, out_frame, out_seq = model.module.init_decoder_hidden(target.size(0))
            else :
                hidden, out_frame, out_seq = model.init_decoder_hidden(target.size(0))


            # forward
            optimizer.zero_grad()

            # output = model(music_seed, pos, dance_seed, i)
            output = model(music_seed, pos, dance_seed, hidden, out_frame, out_seq, epoch)


            # backward
            loss = criterion(output, target)

            loss.backward()

            # update parameters
            optimizer.step()

            current_loss = current_loss + loss.item()

        if epoch == 1000:
            scheduler.step()

        if epoch == 3000:
            scheduler.step()

        epoch_str = "| {0:3.0f} ".format(epoch)[:5]
        perc_str = "({0:3.2f}".format(epoch*100.0 / maxEpochs)[:5]
        error_str = "%) |{0:5.2f}".format(current_loss)[:10] + "|"
        print(epoch_str, perc_str, error_str)


        if (epoch%500 == 0) :
            torch.save({"model": model.state_dict(), "loss" : current_loss}, \
                       f'{data_dir}models/epoch_{epoch}_model_parameters.pth')

if __name__ == '__main__':

    app.run(main)

