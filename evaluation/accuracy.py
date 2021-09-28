import os
import sys
import numpy as np
import torch
from absl import app
from absl import flags
import pickle
import json

import torch.nn.functional as F

import evaluation.classifier as classifier
from evaluation.classifier import Classifier

FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('motion_path',"./temp", "path to normalised 3d keypoints + extension")


def extract_label(name):

    label = -1

    if "BR" in name:
        label = 0
    elif "HO" in name:
        label = 1
    elif "MH" in name:
        label = 2
    elif "KR" in name:
        label = 3
    elif "LH" in name:
        label = 4
    elif "LO" in name:
        label = 5
    elif "PO" in name:
        label = 6
    elif "WA" in name:
        label = 7
    elif "JB" in name:
        label = 8
    elif "JS" in name:
        label = 9

    return label



def get_accuracy(dance, name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = torch.load("./evaluation/checkpoints/final", map_location=torch.device('cpu'))
    classifier.eval()
    classifier.to(device)

    import pdb; pdb.set_trace()

    dance = np.array([np.array(x.cpu()) for x in dance])

    label = torch.cuda.LongTensor([extract_label(name)])if torch.cuda.is_available() else torch.LongTensor([extract_label(name)])

    dance = torch.cuda.FloatTensor(dance) if torch.cuda.is_available() else torch.FloatTensor(dance)

    logits, _ = classifier(dance.unsqueeze(0).to(device))

    loss = F.cross_entropy(logits, torch.LongTensor([label]))

    correct = (torch.max(logits, 1)
                 [1].view(label.size()).data == label.data)

    return loss, correct



def main(_):

    dance_data, labels = load_data(FLAGS.data_dir, 51 )
    loader = prepare_dataloader(dance_data, labels, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = torch.load("./classifier_model", map_location=torch.device('cpu'))

    classifier.eval()

    corrects, avg_loss, size = 0, 0, 0
    for i, batch in enumerate(loader):
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

if __name__ == '__main__':
    app.run(main)
