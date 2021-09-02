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
flags.DEFINE_string('output_dir', "/temp-audios", "path to audio + extension")

SAMPLE_RATE = 48000
HOP_LENGTH = 800
WIN_LENGTH = 1024

def main(_):
    audio_dir = FLAGS.audio_dir
    output_dir = FLAGS.output_dir

    audios = [os.path.join(audio_dir, audio) for audio in os.listdir(audio_dir) if audio.endswith('.mp3')]

    for i in range(len(audios)):

        this_audio_name = audios[i].split("/")[-1].replace(".mp3", "")
        # this_output_dir= os.path.join(output_dir, audio_name)

        this_audio, this_sr = librosa.load(audios[i], sr = SAMPLE_RATE)

        env = calculateAmplitudeEnvelope(this_audio)

        mel = calculateMelDb(this_audio, this_sr)

        mfcc =  calculateMFCC(mel)
        mfcc_delta =  calculateMFCCdelta(mfcc)

        harm, perc = calculateHpss(this_audio)
        Qchroma = calculateQChroma(harm, this_sr)

        onset = calculateOnset(perc, this_sr)

        tempogram = calculateTempogram(onset, this_sr)

        beat = calculateBeat(onset, this_sr)

        l = min(env.shape[1], mfcc.shape[1], mfcc_delta.shape[1],tempogram.shape[1], Qchroma.shape[1], onset.shape[0], beat.shape[1])-1

        concat = np.concatenate((env[:,:l], mfcc[:, :l], mfcc_delta[:, :l], Qchroma[:, :l], onset[None, :l], beat[:, :l], tempogram[:, :l]))
        concat = concat.transpose()

        save_obj(concat, output_dir+"/",  this_audio_name)

        sys.stderr.write('\rextracting %d / %d' % (i + 1, len(audios)))

    sys.stderr.write('\ndone.\n')


def calculateMelDb(audio, sample_rate):
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def calculateHpss(audio):
    audio_harmonic, audio_percussive = librosa.effects.hpss(audio)
    return audio_harmonic, audio_percussive

def calculateMFCC(mel_db):
    mfcc = librosa.feature.mfcc(S=mel_db)
    return mfcc

def calculateMFCCdelta(mfcc):
    mfcc = librosa.feature.delta(mfcc, width=3)
    return mfcc

def calculateChroma(audio_file, sample_rate):
    chroma = librosa.feature.chroma_stft(y=audio_file, sr=sample_rate, )
    return chroma

def calculateQChroma(harm, sample_rate):
    Qchroma = librosa.feature.chroma_stft(y=harm, sr=sample_rate, win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    return Qchroma

def calculateAmplitudeEnvelope(audio_file):
    amplitude_envelope = []

    for i in range(0, len(audio_file), HOP_LENGTH):
        current_frame_ae = max(audio_file[i:i+WIN_LENGTH])
        amplitude_envelope.append(current_frame_ae)

    return np.expand_dims(np.array(amplitude_envelope), axis=0)

def calculateOnset(perc, sample_rate):
    onset = librosa.onset.onset_strength(perc, aggregate=np.median, sr=sample_rate, hop_length=HOP_LENGTH)
    return onset

def calculateTempogram(onset_env, sr):
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    return tempogram

def calculateBeat(onset, sample_rate):
    onset_tempo, onset_beats = librosa.beat.beat_track(onset_envelope=onset, sr=sample_rate,  hop_length=HOP_LENGTH)
    beat = np.zeros(len(onset))
    for idx in onset_beats:
        beat[idx] = 1
    beat = beat.reshape(1, -1)

    return beat

def extract_audio(video_path, save_path):
    subprocess.call(['ffmpeg',  '-y', '-i', video_path, save_path, "-nostats", "-loglevel", "error"])
    # ffmpeg.input(video_path).output(save_path).run()


def find_tempo(audio_file):
    y, sr = librosa.load(audio_file, sr=48000)

    hop_length = 512

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    print("estimated tempo : {:n} beats per minute".format(tempo))


if __name__ == '__main__':
  app.run(main)
