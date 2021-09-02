import os
import ffmpeg
import librosa
from subprocess import Popen, PIPE


def extract_audio(video_path, save_path):

    cmd = ['ffmpeg',  '-y', '-i', video_path, save_path,'-vn', '-sn', '-dn', "-copy_unknown", "-nostats", "-loglevel", "error"]

    # If error
    if run(cmd) :
        print(video_path)

def run(cmd):
    p = Popen(cmd, stderr=PIPE, stdout=PIPE)
    output, errors = p.communicate()
    if p.returncode or errors:
        return 1
    else:
        return 0

def find_tempo(audio_file):
    y, sr = librosa.load(audio_file, sr=44100)

    hop_length = 512

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    print("Estimated Tempo : {:n} beats per minute".format(tempo))


