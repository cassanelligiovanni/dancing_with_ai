# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
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
from dataset import *


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_prefix)
    torch.save(model, save_path)


def eval(dev_data, classifier, device, args):
    classifier.eval()
    corrects, avg_loss, size = 0, 0, 0
    for i, batch in enumerate(dev_data):
        dance, label = map(lambda x: x.to(device), batch)
        dance =  dance.type(torch.cuda.FloatTensor)if torch.cuda.is_available() else  dance.type(torch.FloatTensor)
        logits, _ = classifier(dance)
        loss = F.cross_entropy(logits, label)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(label.size()).data == label.data).sum()
        size += batch[0].shape[0]

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def train(train_data, dev_data, classifier, device, args):
    optimizer = optim.Adam(filter(lambda x: x.requires_grad,
                                  classifier.parameters()), lr=args.lr)
    best_acc = 0
    steps = 0
    classifier.train()
    for epoch in range(1, args.epochs+1):

        for i, batch in enumerate(train_data):
            dance, label = map(lambda x: x.to(device), batch)

            dance =  dance.type(torch.cuda.FloatTensor)if torch.cuda.is_available() else  dance.type(torch.FloatTensor)
            optimizer.zero_grad()
            logits, _ = classifier(dance)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            steps += 1

            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
                train_acc = 100.0 * corrects / batch[0].shape[0]
                print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                               loss.item(),
                                                                               train_acc,
                                                                               corrects,
                                                                               batch[0].shape[0]))


        # evaluate the model on test set at each epoch
        dev_acc = eval(dev_data, classifier, device, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
            save(classifier, args.save_dir, args.save_model, steps)


def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='data/train_1min',
                        help='the directory of dance data')
    parser.add_argument('--valid_dir', type=str, default='data/valid_1min',
                        help='the directory of music feature data')
    parser.add_argument('--save_model', type=str, default='best_model_200dim_50interval_rnn',
                        help='model name')
    parser.add_argument('--save_dir', metavar='PATH', default='checkpoints/')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=51)
    parser.add_argument('--interval', type=int, default=51)

    args = parser.parse_args()

    # TRAIN data
    train_data, train_labels = load_data(args.train_dir, args.interval)
    z = list(zip(train_data, train_labels))
    random.shuffle(z)
    train_data, train_labels = zip(*z)


    train_loader = prepare_dataloader(train_data, train_labels, 32)

    # TEST data
    test_data, test_labels = load_data(args.valid_dir, args.interval)
    z = list(zip(test_data, test_labels))
    random.shuffle(z)
    test_data, test_labels = zip(*z)

    dev_loader = prepare_dataloader(test_data, test_labels, 32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = Classifier()

    for name, parameters in classifier.named_parameters():
        print(name, ':', parameters.size())

    classifier = classifier.to(device)
    train(train_loader, dev_loader, classifier, device, args)
    save(classifier, args.save_dir, "final", 0)



if __name__ == '__main__':
    main()
