import os
import sys
import numpy as np
from absl import app
from absl import flags
import pickle
import json

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('motion_path',"./temp", "path to normalised 3d keypoints + extension")


def main(_):

    data_dir = FLAGS.data_dir
    motion_path = FLAGS.motion_path

    paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith(".json")]

    min_diff = 1000000000
    min_index = 0

    for i, path in enumerate(paths) :
        with open(path) as f:

            sample = json.loads(f.read())
            train = np.array(sample['motion'])

            generated = np.load(motion_path, allow_pickle = True)

            generated = generated.reshape(generated.shape[0],  -1)

            downsampled =np.array([generated[j] for j in range(len(generated)) if j%3==0])

            diff = np.sum(np.abs(train - downsampled))


            if diff < min_diff:
                min_index = i
                min_diff = diff

    print(min_diff)
    print(paths[min_index])


if __name__ == '__main__':
    app.run(main)
