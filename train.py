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

from utils.utils import *
from model import *
from dataset import *

np.random.seed(0)
torch.manual_seed(0)



if __name__ == '__main__':

    INPUT_SIZE = 439

    D_POSE_VEC = 51

    D_MODEL = 200
    N_LAYERS = 1
    N_HEAD = 8
    D_K, D_V = 64, 64
    D_INNER = 1024
    DROPOUT = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    encoder = Encoder(max_seq_len=2878,
                      input_size=INPUT_SIZE,
                      d_word_vec=D_MODEL,
                      n_layers=N_LAYERS,
                      n_head=N_HEAD,
                      d_k=D_K,
                      d_v=D_V,
                      d_model=D_MODEL,
                      d_inner=D_INNER,
                      dropout=DROPOUT)

    decoder = Decoder(input_size=D_POSE_VEC,
                      d_word_vec=D_POSE_VEC,
                      hidden_size=D_INNER,
                      encoder_d_model=D_MODEL,
                      dropout=DROPOUT)


    model = Model(encoder, decoder,
                  condition_step=30,
                  sliding_windown_size=852,
                  lambda_v=0.01,
                  device=device)


    learningRate = 0.0001
    maxEpochs = 3500
    batch_size = 32

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    train_dir = "../gMH/train"

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

        if epoch == 800:
            scheduler.step()

        if epoch == 2000:
            scheduler.step()

        epoch_str = "| {0:3.0f} ".format(epoch)[:5]
        perc_str = "({0:3.2f}".format(epoch*100.0 / maxEpochs)[:5]
        error_str = "%) |{0:5.2f}".format(current_loss)[:10] + "|"
        print(epoch_str, perc_str, error_str)


        if (epoch%100 == 0) :
            torch.save({"model": model.state_dict(), "loss" : current_loss}, \
                       f'./models/epoch_{epoch}_model_parameters.pth')
