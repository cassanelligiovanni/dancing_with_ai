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
flags.DEFINE_string('out_dir',"./temp", "path to normalised 3d keypoints + extension")

train_music = ["BR0", "BR1", "BR2", "BR3", "BR4", "HO0", "HO1", "HO2", "HO3", "HO4", "MH0", "MH1", "MH2", "MH3", "MH4",
               "KR0", "KR1", "KR2", "KR3", "KR4","LH0", "LH1", "LH2", "LH3", "LH4","LO0", "LO1", "LO2", "LO3", "LO4","PO0", "PO1", "PO2", "PO3", "PO4","WA0", "WA1", "WA2", "WA3", "WA4"]

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

    for audio_path in audio_paths:
        plk = np.load(audio_path, allow_pickle = True)
        if plk.shape[0]<min_length :
            min_length = plk.shape[0]

    # Find minimum length
    for test_audio_name in test_music:

        test_motion_names = []
        for motion in motion_names:
            if ("m"+test_audio_name) in motion:
                test_motion_names.append(motion)

        plk = np.load(os.path.join(motion_dir, test_motion_names[0]), allow_pickle = True)
        if plk.shape[0]<min_length :
            min_length = plk.shape[0]

    for train_audio_name in train_music:

        train_motion_names = []
        for motion in motion_names:
            if ("m"+train_audio_name) in motion:
                train_motion_names.append(motion)

        plk = np.load(os.path.join(motion_dir, train_motion_names[0]), allow_pickle = True)
        if plk.shape[0]<min_length :
            min_length = plk.shape[0]

    # Chunk and save
    for test_audio_name in test_music:

        test_motion_names = []
        for motion in motion_names:
            if ("m"+test_audio_name) in motion:
                test_motion_names.append(motion)

        chuck_and_save(test_motion_names[0], min_length, training=False )


    for train_audio_name in train_music:

        train_motion_names = []
        for motion in motion_names:
            if ("m"+train_audio_name) in motion:
                train_motion_names.append(motion)

        for train_motion_name in train_motion_names:
            chuck_and_save(train_motion_name, min_length, training=True )

        # chuck_and_save(train_motion_names[0], min_length, training=True )

    # sys.stderr.write('\rextracting %d / %d' % (i + 1, len(audio_names)))

    # sys.stderr.write('\ndone.\n')


def align(musics, dances):
    assert len(musics) == len(dances), \
        'the number of audios should be equal to that of videos'
    new_musics=[]
    new_dances=[]

    new_musics.append([musics[j] for j in range(len(musics)) if j%3==0])
    new_musics.append([musics[j] for j in range(len(musics)) if j%3==1])
    new_musics.append([musics[j] for j in range(len(musics)) if j%3==2])

    new_dances.append([dances[j] for j in range(len(musics)) if j%3==0])
    new_dances.append([dances[j] for j in range(len(musics)) if j%3==1])
    new_dances.append([dances[j] for j in range(len(musics)) if j%3==2])

    return new_musics, new_dances



def chuck_and_save(name, max_length, training):

    audio = np.load(FLAGS.audio_dir+name, allow_pickle=True)
    motion = np.load(FLAGS.motion_dir+name, allow_pickle=True )

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

        downsmpl_music, downsmpl_motion = align(new_audio.tolist(), new_motion.tolist())

        for f in range(3):

            sequence = {}
            sequence["id"] = name.split(".")[0]+"_"+str(i)+"_"+str(f)
            sequence["music"] = downsmpl_music[f]
            sequence["motion"] = downsmpl_motion[f]

            # if(i==4):
            #     import pdb; pdb.set_trace()

            if (training):
                save_json(sequence, FLAGS.out_dir+"train/"+sequence['id']+".json")
            else :
                save_json(sequence, FLAGS.out_dir+"test/"+sequence['id']+".json")

if __name__ == '__main__':
    app.run(main)
