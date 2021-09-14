import os
import sys
from absl import app
from absl import flags
from utils.extract_audio import *
import librosa


FLAGS = flags.FLAGS
flags.DEFINE_string('videos_dir', "/temp-videos", "path to video + extension")
flags.DEFINE_string('save_dir', "/temp-audios", "path to audio + extension")

def main(_):

    videos_dir = FLAGS.videos_dir
    video_names = os.listdir(videos_dir)
    save_dir = FLAGS.save_dir

    # Take all the .mp4 files
    videos_path = [name for name in video_names if name.endswith(".mp4")]

    for i  in range(len(videos_path)):

    # for i in range(1):

        name = videos_path[i]
        path = os.path.join(videos_dir, name)

        audio_name = os.path.splitext(name)[0] + ".mp3"

        save_path = os.path.join(save_dir, audio_name.replace("c01", "cAll"))
        extract_audio(path, save_path)
        sys.stderr.write('\rextracting %d / %d' % (i + 1, len(videos_path)))

    sys.stderr.write('\ndone.\n')

    # save_path = FLAGS.save_path
    # extract_audio(video_path, save_path)

    # IPython.display.Audio(data=y, rate=sr)

if __name__ == '__main__':
  app.run(main)
