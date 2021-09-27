import numpy as np

from absl import app
from absl import flags
import pickle

import json

def save_json(file, path) :

    with open(path, 'w') as f:

        json.dump(file, f)





def loss_function(input,target):

    criterion = nn.MSELoss()
    loss = criterion(input, target)

    return loss

def save_obj(obj, directory, name ):
    with open(directory + name +  '.pkl', 'wb') as f:
        pickle.dump(obj, f, 4)


