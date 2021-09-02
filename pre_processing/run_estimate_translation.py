import os
import numpy as np
from absl import app
from absl import flags
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string('keypoints3d_path',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('normalised3d_path',"./temp", "path to normalised 3d keypoints + extension")

def main(_):

    all_keypoints = [f for f in os.listdir(FLAGS.keypoints3d_path) if f.endswith('.pkl')]

    estimate_translation(all_keypoints)


def save_obj(obj, name, directory):
    with open(directory + name, 'wb') as f:
        pickle.dump(obj, f, 4)


def estimate_translation(dataset):
    # For all the dataset set
    for j in range(len(dataset)):

        this_seq_path = FLAGS.keypoints3d_path +  dataset[j]
        this_seq = np.load(this_seq_path, allow_pickle=True)['keypoints3d_optim']

        [n_frames, n_joints, n_dim] = this_seq.shape

        normalised_seq = np.zeros((n_frames, n_joints+1, n_dim))

        normalised_seq = this_seq

        # for i in range(this_seq.shape[0]):
        #     this_frame = this_seq[i]

        #     # Take left and right hip
        #     left_hip = this_frame[11]
        #     right_hip = this_frame[12]

        #     # and calculate the mean
        #     pelvis = (left_hip +  right_hip)/2

        #     # Normalise all the other joint with pelvis at the 0 center
        #     normalised_frame = this_frame - pelvis

        #     normalised_seq[i, :-1] = normalised_frame
        #     normalised_seq[i, n_joints] = pelvis

        path_to = FLAGS.normalised3d_path

        # Save to .pkl file
        save_obj(normalised_seq, dataset[j], path_to)


if __name__ == '__main__':
    app.run(main)
