import os
import sys
import numpy as np
from absl import app
from absl import flags
import pickle
import json
from classifier import Classifier
from dataset import *
from dataset import *

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('motion_path',"./temp", "path to normalised 3d keypoints + extension")



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
