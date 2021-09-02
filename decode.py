import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
import torch.nn as nn
from torch.nn import Module
import torchvision
from torch.nn.utils.rnn import pack_sequence

from utils import *
from DAE import *


from absl import app
from absl import flags
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string('model_parameters',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('encoded_path',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('decoded_dir',"./temp", "path to encoded 3d keypoints + extension")
flags.DEFINE_string('decoded_name',"temp", "path to encoded 3d keypoints + extension")

def main(_):

    eps = 1e-8

    decoder = Decoder()

    motion = np.load(FLAGS.encoded_path, allow_pickle=True)

    parameters = torch.load(FLAGS.model_parameters)
    decoder.load_state_dict(parameters['decoder'])
    decoder.eval()

    max_value = parameters['max_value']
    mean_pose = parameters['mean_pose']

    output = decoder.forward(motion)
    output = convert_back_to_3d(output, max_value, mean_pose).detach().numpy()
    output = output.reshape(motion.shape[0], 18, 3)

    output_name = FLAGS.encoded_path.split("/")[-1].replace(".pkl", "")

    save_obj(output, FLAGS.decoded_dir, output_name)

    print("###### DECODING Finished  ######")


if __name__ == '__main__':

    app.run(main)

