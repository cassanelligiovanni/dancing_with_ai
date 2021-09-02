import os
import sys
import numpy as np

from absl import app
from absl import flags
import pickle

from encode import encode_motion

FLAGS = flags.FLAGS
flags.DEFINE_string('model_parameters',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('to_encode_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('encoded_dir',"./temp", "path to encoded 3d keypoints + extension")


def main(_):

    model_parameters = FLAGS.model_parameters
    to_encode_dir = FLAGS.to_encode_dir
    encoded_dir = FLAGS.encoded_dir

    ## TRAINING DATA
    print("#### ENCODING Training Set #####")

    train_encoded_dir = os.path.join(encoded_dir, "train/")

    training_dir = os.path.join(to_encode_dir, "train")
    training = [os.path.join(training_dir, motion) for motion in os.listdir(training_dir) if motion.endswith(".pkl")]
    training_names = [motion.replace(".pkl", "") for motion in os.listdir(training_dir) if motion.endswith(".pkl")]

    for i in range(len(training)):

        this_to_encode_path = training[i]
        this_encoded_name = training_names[i]

        encode_motion(this_to_encode_path, model_parameters, train_encoded_dir, this_encoded_name)
        sys.stderr.write('\rencoding %d / %d' % (i + 1, len(training)))

    sys.stderr.write('\ndone.\n')



    ## TESTING DATA

    print("#### ENCODING Testing Dataset #####")

    test_encoded_dir = os.path.join(encoded_dir, "test/")

    testing_dir = os.path.join(to_encode_dir, "test")
    testing = [os.path.join(testing_dir, motion) for motion in os.listdir(testing_dir) if motion.endswith(".pkl")]
    testing_names = [motion.replace(".pkl", "") for motion in os.listdir(testing_dir) if motion.endswith(".pkl")]

    for i in range(len(testing)):

        this_to_encode_path = testing[i]
        this_encoded_name = testing_names[i]

        encode_motion(this_to_encode_path, model_parameters, test_encoded_dir, this_encoded_name)
        sys.stderr.write('\rencoding %d / %d' % (i + 1, len(testing)))

    sys.stderr.write('\ndone.\n')





if __name__ == '__main__':

    app.run(main)

