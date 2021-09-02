import os
import sys
import numpy as np
from absl import app
from absl import flags
import pickle
from utils.utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string('motion_dir',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('audio_dir',"./temp", "path to normalised 3d keypoints + extension")

test_music = ["BR5", "HO5", "JB5", "JS5", "KR5", "LH5", "LO5", "MH5", "PO5", "WA5"]

def main(_):

    motion_dir = FLAGS.motion_dir
    audio_dir = FLAGS.audio_dir

    motion_names = [fname for fname in os.listdir(motion_dir) if fname.endswith('.pkl')]
    motion_paths = [os.path.join(motion_dir, fname) for fname in motion_names ]
    audio_names = [fname for fname in os.listdir(audio_dir) if fname.endswith('.pkl')]
    audio_paths = [os.path.join(audio_dir, fname) for fname in audio_names ]

    test_audio_names = [name+".plk" for name in test_music]

    # Find minium length of sequence
    min_length = 10e9
    lengths = []
    for path in motion_paths:
        plk = np.load(path, allow_pickle = True)
        lengths.append(plk.shape[0])
        if plk.shape[0]<min_length :
            min_length = plk.shape[0]


    for audio_path in audio_paths:
        plk = np.load(audio_path, allow_pickle = True)
        if plk.shape[0]<min_length :

            min_length = plk.shape[0]



    for test_audio_name in test_music:

        test_motion_names = []
        for motion in motion_names:
            if ("m"+test_audio_name) in motion:
                test_motion_names.append(motion)

        chuck_and_save(test_motion_names[0], min_length, training=False )


    for i, name in enumerate(audio_names):

        if not any(x in name for x in test_music):
            chuck_and_save(name, min_length, True)

        sys.stderr.write('\rextracting %d / %d' % (i + 1, len(audio_names)))

    sys.stderr.write('\ndone.\n')






def chuck_and_save(name, max_length, training):

    audio = np.load("./data/audio_features/"+name, allow_pickle=True)
    motion = np.load("./data/motions/"+name, allow_pickle=True )

    # [n_frames x 54(17*3)]
    motion = motion.reshape(motion.shape[0], (motion.shape[1]*motion.shape[2]))

    l_audio, n_audio_features = audio.shape
    l_motion, n_motion_features = motion.shape

    l_sequence = min(l_audio, l_motion)
    n_chunk = int(np.ceil(l_sequence/max_length))

    for i in range(n_chunk-1):

        start = i*max_length
        end = (i+1)*max_length

        new_audio = np.zeros((max_length, n_audio_features))
        new_motion = np.zeros((max_length, n_motion_features))

        l_new_audio = audio[start:end].shape[0]
        new_audio[:l_new_audio, :] = audio[start:end, :]

        l_new_motion = motion[start:end].shape[0]
        new_motion[:l_new_motion, :] = motion[start:end, :]

        sequence = {}
        sequence["id"] = name.split(".")[0]+"_"+str(i)
        sequence["music"] = new_audio.tolist()
        sequence["motion"] = new_motion.tolist()

        # if(i==4):
        #     import pdb; pdb.set_trace()

        if (training):
            save_json(sequence, "./data/dataset/train/"+sequence['id']+".json")
        else :
            save_json(sequence, "./data/dataset/test/"+sequence['id']+".json")

if __name__ == '__main__':
    app.run(main)
