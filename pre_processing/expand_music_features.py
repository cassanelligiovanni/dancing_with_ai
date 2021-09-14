import os
import subprocess
import sys
from absl import app
from absl import flags
from utils.extract_audio import *
import librosa
from utils.utils import *
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FLAGS = flags.FLAGS
flags.DEFINE_string('audio_dir', "/temp-videos", "path to video + extension")
flags.DEFINE_string('motion_dir', "/temp-videos", "path to video + extension")
flags.DEFINE_string('output_dir', "/temp-audios", "path to audio + extension")

SAMPLE_RATE = 48000
HOP_LENGTH = 800
WIN_LENGTH = 1024

def main(_):

    motion_dir = FLAGS.motion_dir
    audio_dir = FLAGS.audio_dir
    output_dir = FLAGS.output_dir

    audios = [os.path.join(audio_dir, audio) for audio in os.listdir(audio_dir) if audio.endswith('.pkl')]
    motions = [os.path.join(motion_dir, motion) for motion in os.listdir(motion_dir) if motion.endswith('.pkl')]

    for i in range(len(motions)):

        this_name = motions[i].split("/")[-1].replace(".pkl", "")
        this_music_name = this_name.split("_")[4]
        this_music_path = audio_dir + this_music_name + ".pkl"

        this_motion = np.load(motions[i], allow_pickle=True)
        this_motion_length = this_motion.shape[0]

        this_music = np.load(this_music_path, allow_pickle=True)[:this_motion_length]


        save_obj(this_music, output_dir+"/",  this_name)

        sys.stderr.write('\rextracting %d / %d' % (i + 1, len(motions)))

    sys.stderr.write('\ndone.\n')



if __name__ == '__main__':
  app.run(main)

