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

import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('d_model',"240", "path to normalised 3d keypoints + extension")
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

WANDB_API_KEY ="f29aca38281e9a7657e6661c5684aa35fecf37ce"


def train(maxEpochs, loader, optimizer, criterion, device, encoder, decoder, model):

    steps = 0
    print(" ______________ ______")
    print("|     Epoch    | RMSE |")
    print("|------------  |------|")
    for epoch in range(1, maxEpochs+1):

        calculate_loss = False
        current_loss = 0

        model.train()

        for i, batch in enumerate(loader):

            steps += batch.shape[0]

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


        train_log(current_loss, steps, epoch)

        epoch_str = "| {0:3.0f} ".format(epoch)[:5]
        perc_str = "({0:3.2f}".format(epoch*100.0 / maxEpochs)[:5]
        error_str = "%) |{0:5.2f}".format(current_loss)[:10] + "|"
        print(epoch_str, perc_str, error_str)

       # Save checkpoints
        if (epoch%500 == 0) :
            torch.save({"model": model.state_dict(), "loss" : current_loss}, \
                       f'{data_dir}models/final_{epoch}_model_parameters.pth')




def main(_):

    wandb.init(entity="cassanelligiovanni", project="dissertation")
    config = wandb.config

    config.data_dir = FLAGS.data_dir
    train_dir = config.data_dir + "dataset/train"

    # Model Parameters
    config.d_model = int(FLAGS.d_model)
    config.n_layers = int(FLAGS.n_layers)
    config.n_heads = int(FLAGS.n_heads)
    config.inner_d = int(FLAGS.inner_d)
    config.MUSIC_SIZE = 439
    config.DANCE_SIZE = 51
    config.D_K, config.D_V = 64, 64

    # Training  Hyper-parameters
    config.learningRate = 0.0001
    config.lambda_v = 0.01
    config.maxEpochs = 10
    config.batch_size = 16
    config.DROPOUT = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(max_seq_len=142,
                      music_size=config.MUSIC_SIZE,
                      n_layers=config.n_layers,
                      n_head=config.n_heads,
                      d_k=config.D_K,
                      d_v=config.D_V,
                      d_model=config.d_model,
                      d_inner=config.inner_d,
                      dropout=config.DROPOUT)

    decoder = Decoder(motion_size=config.DANCE_SIZE,
                      d_emb=config.DANCE_SIZE,
                      hidden_size=config.inner_d,
                      dropout=config.DROPOUT)


    model = Model(encoder, decoder,
                  condition_step=10,
                  lambda_v=config.lambda_v,
                  device=device)

    wandb.watch(model, log="all")

    music_data, dance_data = load_data(train_dir)
    loader = prepare_dataloader(music_data, dance_data, config.batch_size)

    optimizer = optim.Adam(filter(
        lambda x: x.requires_grad, model.parameters()), lr=config.learningRate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    criterion = nn.L1Loss()
    updates = 0

    model = nn.DataParallel(model).to(device) if torch.cuda.is_available() else model.to(device)

    train(config.maxEpochs, loader, optimizer, criterion,  device, encoder, decoder, model)

    torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')

if __name__ == '__main__':

    app.run(main)

