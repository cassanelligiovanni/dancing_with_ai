import os

from utils import *
from DAE import *
from encode import encode_motion

from absl import app
from absl import flags
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string('model_parameters',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('to_encode_path',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('encoded_dir',"./temp", "path to encoded 3d keypoints + extension")
flags.DEFINE_string('encoded_name',"temp", "path to encoded 3d keypoints + extension")


def main(_):

    to_encode_path = FLAGS.to_encode_path
    model_parameters = FLAGS.model_parameters
    encoded_dir = FLAGS.encoded_dir
    encoded_name = FLAGS.encoded_name

    encode_motion(to_encode_path, model_parameters, encoded_dir, encoded_name)

    print("###### ENCODED Finished  ######")

if __name__ == '__main__':

    app.run(main)

