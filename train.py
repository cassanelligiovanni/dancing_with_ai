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

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('d_model',"300", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('n_layers',"2", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('n_heads',"8", "path to normalised 3d keypoints + extension")
flags.DEFINE_string('inner_d',"1024", "path to normalised 3d keypoints + extension")

# Set random seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def main(_):

    data_dir = FLAGS.data_dir
    train_dir = data_dir + "dataset/train"

    # Model Parameters
    d_model = int(FLAGS.d_model)
    n_layers = int(FLAGS.n_layers)
    n_heads = int(FLAGS.n_heads)
    inner_d = int(FLAGS.inner_d)
    MUSIC_SIZE = 439
    DANCE_SIZE = 51
    D_K, D_V = 64, 64

    # Training  Hyper-parameters
    learningRate = 0.0001
    maxEpochs = 20000
    batch_size = 16
    DROPOUT = 0.1

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
                      d_emb=DANCE_SIZE,
                      hidden_size=inner_d,
                      dropout=DROPOUT)


    model = Model(encoder, decoder,
                  condition_step=10,
                  lambda_v=0.01,
                  device=device)

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


    music_data, dance_data = load_data(train_dir)
    loader = prepare_dataloader(music_data, dance_data, batch_size)

    optimizer = optim.Adam(filter(
        lambda x: x.requires_grad, model.parameters()), lr=learningRate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    criterion = nn.L1Loss()
    updates = 0

    model = nn.DataParallel(model).to(device) if torch.cuda.is_available() else model.to(device)

    print(" ______________ ______")
    print("|     Epoch    | RMSE |")
    print("|------------  |------|")
    for epoch in range(1, maxEpochs+1):

        calculate_loss = False
        current_loss = 0

        model.train()

        for i, batch in enumerate(loader):

            music, pos, dance = map(lambda x: x.to(device), batch)
            target = dance[:, 1:]
            music = music[:, :-1]
            pos = pos[:, :-1]
            dance = dance[:, :-1]

            #Initialise
            if torch.cuda.is_available() :
                hidden, initial_frame, initial_seq = model.module.initialise_decoder(target.size(0))
            else :
                hidden, initial_frame, initial_seq = model.initialise_decoder(target.size(0))


            optimizer.zero_grad()

            # Forward
            output = model(music, pos, dance, hidden, initial_frame, initial_seq, epoch)

            # Backpropagation
            loss = criterion(output, target)
            loss.backward()

            # Update parameters
            optimizer.step()

        # Track loss
            current_loss = current_loss + loss.item()

        if epoch == 1000:
            scheduler.step()

        if epoch == 2000:
            scheduler.step()

        epoch_str = "| {0:3.0f} ".format(epoch)[:5]
        perc_str = "({0:3.2f}".format(epoch*100.0 / maxEpochs)[:5]
        error_str = "%) |{0:5.2f}".format(current_loss)[:10] + "|"
        print(epoch_str, perc_str, error_str)

       # Save checkpoints
        if (epoch%500 == 0) :
            torch.save({"model": model.state_dict(), "loss" : current_loss}, \
                       f'{data_dir}models/final_{epoch}_model_parameters.pth')

if __name__ == '__main__':

    app.run(main)

